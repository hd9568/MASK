import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from itertools import chain


class DataType(Enum):
    u64 = 0
    i64 = 1
    u32 = 2
    i32 = 3
    f64 = 4
    f32 = 5
    f16 = 6


class Param:
    def __init__(self, name: str, type_str: str):
        self.name = name

        # Use regex to match different type_str patters. 3 patterns in total:
        # 1. plain, e.g. "int", "float", set is_pointer to False and dims to []
        # 2. pointer, e.g. "int*", "float*", set is_pointer to True
        # 3. array, e.g. "int[10]", "float[10][20]", set dims to [10], [10, 20], respectively
        # whitespace is allowed in type_str, only one level of pointer is allowed
        match = re.match(r"(\w+)\s*(\*?)\s*((?:\[\d+]\s*)*)", type_str)
        if match is None:
            raise ValueError(f"invalid type_str: {type_str}")

        base_type_str = match.group(1)
        self.dtype = self.base_type_from_str(base_type_str)
        self.is_pointer = bool(match.group(2))
        dims_str = match.group(3)
        dim_matches = re.findall(r"\[(\d+)]", dims_str)
        self.dims = list(map(int, dim_matches))

        if self.is_pointer and self.dims:
            raise ValueError("pointer and dims cannot be specified at the same time")

    @property
    def base_type(self) -> str:
        match self.dtype:
            case DataType.u64:
                ts = "uint64_t"
            case DataType.i64:
                ts = "int64_t"
            case DataType.u32:
                ts = "uint32_t"
            case DataType.i32:
                ts = "int32_t"
            case DataType.f64:
                ts = "double"
            case DataType.f32:
                ts = "float"
            case DataType.f16:
                ts = "half"
            case _:
                raise ValueError(f"unknown dtype: {self.dtype}")

        return ts

    @staticmethod
    def base_type_from_str(base_type_str: str) -> DataType:
        match base_type_str:
            case "u64" | "uint64_t" | "unsigned long long" | "unsigned long long int":
                return DataType.u64
            case "i64" | "int64_t" | "long long" | "long long int":
                return DataType.i64
            case "u32" | "uint32_t" | "unsigned int":
                return DataType.u32
            case "i32" | "int32_t" | "int":
                return DataType.i32
            case "f64" | "double":
                return DataType.f64
            case "f32" | "float":
                return DataType.f32
            case "f16" | "half" | "__half":
                return DataType.f16
            case _:
                raise ValueError(f"unknown base type: {base_type_str}")

    def gen_decl(self) -> str:
        if self.dims:
            return f"{self.base_type} {self.name}[{']['.join(map(str, self.dims))}];"
        elif self.is_pointer:
            return f"{self.base_type}* {self.name};"
        else:
            return f"{self.base_type} {self.name};"

    def gen_array_decl(self, length: int) -> str:
        if self.dims:
            return f"{self.base_type} {self.name}[{length}][{']['.join(map(str, self.dims))}];"
        elif self.is_pointer:
            return f"{self.base_type}* {self.name}[{length}];"
        else:
            return f"{self.base_type} {self.name}[{length}];"

    def gen_param_use(self) -> str:
        if self.dims:
            return f"{self.base_type} {self.name}[{']['.join(map(str, self.dims))}]"
        elif self.is_pointer:
            return f"{self.base_type}* {self.name}"
        else:
            return f"{self.base_type} {self.name}"

    def gen_array_param_use(self, length: int) -> str:
        if self.dims:
            return f"{self.base_type} {self.name}[{length}][{']['.join(map(str, self.dims))}]"
        elif self.is_pointer:
            return f"{self.base_type}* {self.name}[{length}]"
        else:
            return f"{self.base_type} {self.name}[{length}]"


class Params:
    HW_IDX: OrderedDict[str, Param] = OrderedDict(
        [
            ("block_idx_x", Param("block_idx_x", "int")),
            ("lane_id", Param("lane_id", "int")),
        ]
    )

    def __init__(
        self,
        sw_idx: OrderedDict[str, Param] = OrderedDict(),
        const: OrderedDict[str, Param] = OrderedDict(),
        gmem: OrderedDict[str, Param] = OrderedDict(),
        smem: OrderedDict[str, Param] = OrderedDict(),
        reg: OrderedDict[str, Param] = OrderedDict(),
        is_global: bool = False,
    ):
        self.sw_idx: OrderedDict[str, Param] = OrderedDict(sw_idx)
        self.const: OrderedDict[str, Param] = OrderedDict(const)
        self.gmem: OrderedDict[str, Param] = OrderedDict(gmem)
        self.smem: OrderedDict[str, Param] = OrderedDict(smem)
        self.reg: OrderedDict[str, Param] = OrderedDict(reg)
        self.is_global: bool = is_global

    def gen_param_list(self) -> str:
        param_list = f"(\n{',\n'.join('  ' + pt + ' ' + pn for pn, pt in chain(
            map(lambda p: ("_H_" + p[0], p[1].base_type), self.HW_IDX.items()) if not self.is_global else [],
            map(lambda p: ("_I_" + p[0], p[1].base_type), self.sw_idx.items()),
            map(lambda p: ("_C_" + p[0], p[1].base_type), self.const.items()),
            map(lambda p: ("_G_" + p[0], p[1].base_type), self.gmem.items()),
            map(lambda p: ("_S_" + p[0], p[1].base_type), self.smem.items()),
            map(lambda p: ("_R_" + p[0], p[1].base_type), self.reg.items()),
        ))}\n)"
        return param_list

    def gen_arg_list(self, **kwargs) -> str:
        args = [
            kwargs[pn]
            for pn in chain(
                self.HW_IDX if not self.is_global else [],
                self.sw_idx,
                self.const,
                self.gmem,
                self.smem,
                self.reg,
            )
        ]
        return ", ".join(map(str, args))

    def default_kwargs(self):
        return {
            **{pn: "_H_" + pn for pn in self.HW_IDX if not self.is_global},
            **{pn: "_I_" + pn for pn in self.sw_idx},
            **{pn: "_C_" + pn for pn in self.const},
            **{pn: "_G_" + pn for pn in self.gmem},
            **{pn: "_S_" + pn for pn in self.smem},
            **{pn: "_R_" + pn for pn in self.reg},
        }

    def gen_default_arg_list(self) -> str:
        return self.gen_arg_list(**self.default_kwargs())
