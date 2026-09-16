# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
from pathlib import Path
import tempfile
from typing import List, Optional

import numpy as np
import torch
from biotite.structure import AtomArray

from opendde.data.inference.input_validation import (
    validate_inference_seed,
    validate_sample_name,
)
from opendde.data.utils import save_structure_cif
from opendde.utils.file_io import save_json


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing
    unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """

    def _rounded_copy(value):
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.detach().cpu().numpy()
            return (
                np.round(value, 2)
                if np.issubdtype(value.dtype, np.floating)
                else value.copy()
            )
        if isinstance(value, np.ndarray):
            return (
                np.round(value, 2)
                if np.issubdtype(value.dtype, np.floating)
                else value.copy()
            )
        if isinstance(value, list):
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.floating):
                return list(np.round(array, 2))
            return list(value)
        if isinstance(value, dict):
            return {key: _rounded_copy(item) for key, item in value.items()}
        return value

    # Build a detached serialization tree. Dumping output is an I/O operation
    # and must not delete keys or replace tensors in the caller's prediction.
    return {
        key: _rounded_copy(value)
        for key, value in full_confidence_dict.items()
        if key not in {"atom_coordinate", "atom_is_polymer"}
    }


class DataDumper:
    """
    Class for dumping prediction data, including structure coordinates and confidence scores.

    Args:
        base_dir (str): Base directory for saving dumped data.
        need_atom_confidence (bool): Whether to save detailed atom-level confidence data.
        sorted_by_ranking_score (bool): Accepted for compatibility; filenames always
            use the original diffusion sample index.
    """

    def __init__(
        self,
        base_dir: str,
        need_atom_confidence: bool = False,
        sorted_by_ranking_score: bool = True,
        compress_full_confidence: bool = False,
    ) -> None:
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence
        self.sorted_by_ranking_score = sorted_by_ranking_score
        self.compress_full_confidence = compress_full_confidence

    @staticmethod
    def _write_full_data_atomic(path: Path, data: dict, *, compressed: bool) -> None:
        """Publish one confidence file atomically without pickle-backed arrays."""
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            if compressed:
                arrays = {key: np.asarray(value) for key, value in data.items()}
                object_keys = [
                    key for key, value in arrays.items() if value.dtype.hasobject
                ]
                if object_keys:
                    raise TypeError(
                        "NPZ full confidence requires primitive arrays; "
                        f"object values found for {object_keys}."
                    )
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
            else:
                save_json(data, temporary, indent=None)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def dump(
        self,
        group_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            group_name (str): Optional output grouping name.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
        """
        dump_dir = self._get_dump_dir(group_name, pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
        )

    def _get_dump_dir(self, group_name: str, sample_name: str, seed: int) -> str:
        """
        Generate the job directory path and validate output path components.
        """
        sample_name = validate_sample_name(sample_name)
        if group_name:
            group_name = validate_sample_name(group_name)
        validate_inference_seed(seed, location="output seed")
        dump_dir = os.path.join(self.base_dir, group_name, sample_name)
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
    ):
        """
        Dump raw predictions from the model.

        Args:
            pred_dict (dict): Prediction results.
            dump_dir (str): Directory where to save the predictions.
            pdb_id (str): PDB ID or sample name.
            atom_array (AtomArray): Reference atom array for structure formatting.
            entity_poly_type (dict[str, str]): Dictionary mapping entity IDs to their polymer types.
            seed (int): Random seed used for the prediction.
        """
        seed = validate_inference_seed(seed, location="output seed")
        # Dump structure, retaining atom pLDDT as B-factors even when detailed
        # confidence JSON output is disabled.
        b_factor = None
        if "full_data" in pred_dict:
            all_atom_plddt = []
            # len(pred_dict["full_data"]) == N_sample
            for each_sample_dict in pred_dict["full_data"]:
                if "atom_plddt" in each_sample_dict:
                    # atom_plddt.shape == [N_atom]
                    atom_plddt = each_sample_dict["atom_plddt"]
                    if atom_plddt.dtype == torch.bfloat16:
                        atom_plddt = atom_plddt.to(torch.float32)
                    all_atom_plddt.append(atom_plddt.cpu().numpy() * 100.0)

            if len(all_atom_plddt) == len(pred_dict["full_data"]):
                b_factor = all_atom_plddt
        self._save_structure(
            pred_coordinates=pred_dict["coordinate"],
            prediction_save_dir=os.path.join(dump_dir, "models"),
            sample_name=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
            b_factor=b_factor,
        )
        self._save_confidence(data=pred_dict, prediction_save_dir=dump_dir, seed=seed)

    def _save_structure(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
        b_factor: Optional[List[np.ndarray]] = None,
    ):
        """
        Save predicted structures to CIF files.

        Args:
            pred_coordinates (torch.Tensor): Predicted coordinates [N_sample, N_atom, 3].
            prediction_save_dir (str): Directory where to save the structures.
            sample_name (str): Sample name.
            atom_array (AtomArray): Template atom array.
            entity_poly_type (dict[str, str]): Entity polymer types.
            seed (int): Prediction seed.
            b_factor (Optional[List[np.ndarray]]): Predicted LDDT scores to be saved as B-factors.
        """
        assert atom_array is not None
        os.makedirs(prediction_save_dir, exist_ok=True)
        for idx, coordinates in enumerate(pred_coordinates):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"seed-{seed}_sample-{idx}_model.cif",
            )
            output_atom_array = atom_array
            if b_factor is not None:
                # b_factor.shape == [N_sample, N_atom]
                # The dumper is an I/O boundary; annotating the caller's
                # AtomArray makes a later reuse observe the last sample's pLDDT.
                output_atom_array = atom_array.copy()
                output_atom_array.set_annotation("b_factor", np.round(b_factor[idx], 2))

            save_structure_cif(
                atom_array=output_atom_array,
                pred_coordinate=coordinates,
                output_fpath=output_fpath,
                entity_poly_type=entity_poly_type,
                pdb_id=sample_name,
            )

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        seed: int,
    ):
        """
        Save confidence data to JSON files.

        Args:
            data (dict): Prediction results containing confidence scores.
            prediction_save_dir (str): Directory where to save the files.
            seed (int): Prediction seed.
        """
        summary_dir = os.path.join(prediction_save_dir, "summary_confidences")
        os.makedirs(summary_dir, exist_ok=True)
        full_data_dir = os.path.join(prediction_save_dir, "full_data")
        if self.need_atom_confidence:
            os.makedirs(full_data_dir, exist_ok=True)
        for idx, summary in enumerate(data["summary_confidence"]):
            output_fpath = os.path.join(
                summary_dir,
                f"seed-{seed}_sample-{idx}_summary_confidences.json",
            )
            save_json(summary, output_fpath, indent=4)
            prefix = f"seed-{seed}_sample-{idx}_full_data"
            json_path = Path(full_data_dir) / f"{prefix}.json"
            npz_path = Path(full_data_dir) / f"{prefix}.npz"
            if self.need_atom_confidence:
                selected_path, stale_path = (
                    (npz_path, json_path)
                    if self.compress_full_confidence
                    else (json_path, npz_path)
                )
                self._write_full_data_atomic(
                    selected_path,
                    get_clean_full_confidence(data["full_data"][idx]),
                    compressed=self.compress_full_confidence,
                )
                stale_path.unlink(missing_ok=True)
            else:
                json_path.unlink(missing_ok=True)
                npz_path.unlink(missing_ok=True)
