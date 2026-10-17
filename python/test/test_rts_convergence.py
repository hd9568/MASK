from __future__ import annotations

import unittest

from sparsene.format.format import (
    Direction,
    Format,
    csr_atomic_format,
    df_axis,
    dv_axis,
    sv_axis,
)
from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.formats.Spaden import BIT_BSR_FORMAT
from sparsene.transform.rts import derive_rts


class RtsConvergenceTest(unittest.TestCase):
    def test_row_row_col_col_hierarchy_terminates(self) -> None:
        format_ = Format(
            name="balanced-row-group-csr",
            axes=[
                sv_axis("X_g", Direction.ROW, length="M"),
                dv_axis("X_i", Direction.ROW, length="GROUP_ROW_CAP"),
                df_axis("Y_o", Direction.COL, length="K_DIV_BLK_K"),
                df_axis("Y_i", Direction.COL, length="BLK_K"),
            ],
            child=csr_atomic_format("X_i", "Y_i"),
        )

        sequence = derive_rts(format_).sequence

        self.assertLessEqual(len(sequence), 8)
        self.assertEqual(sum(item.name == "split" for item in sequence), 2)
        self.assertEqual(sum(item.name == "swap" for item in sequence), 0)
        self.assertEqual(sequence[-1].name, "csrize")

    def test_existing_me_tcf_sequence_is_unchanged(self) -> None:
        sequence = derive_rts(ME_TCF_FORMAT).sequence
        self.assertEqual(
            [(item.name, item.axes) for item in sequence],
            [
                ("split", (0,)),
                ("swap", (1, 2)),
                ("sparsify", (1,)),
                ("spawn", ()),
                ("split", (0,)),
                ("spawn", ()),
                ("swap", (0, 1)),
                ("cooize", ()),
            ],
        )

    def test_existing_alternating_bit_bsr_sequence_is_unchanged(self) -> None:
        sequence = derive_rts(BIT_BSR_FORMAT).sequence
        self.assertEqual(
            [(item.name, item.axes) for item in sequence],
            [
                ("split", (0,)),
                ("swap", (1, 2)),
                ("split", (1,)),
                ("swap", (2, 3)),
                ("sparsify", (1,)),
                ("spawn", ()),
                ("mcoize", ()),
            ],
        )


if __name__ == "__main__":
    unittest.main()
