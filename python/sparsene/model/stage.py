from enum import Enum
from itertools import chain
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import networkx as nx

HW_IDX_PARAMS: OrderedDict[str, str] = OrderedDict([
    ("block_idx_x", "int"),
    ("lane_id", "int"),
])


@dataclass
class SourceTemplate:
    sw_idx_params: OrderedDict[str, str]
    const_params: OrderedDict[str, str]
    gmem_params: OrderedDict[str, str]
    smem_params: OrderedDict[str, str]
    reg_params: OrderedDict[str, str]
    body: str

    def gen_func_param_list(self) -> str:
        func_param_list = f"(\n{',\n'.join('  ' + pt + ' ' + pn for pn, pt in chain(
            HW_IDX_PARAMS.items(),
            self.sw_idx_params.items(),
            self.const_params.items(),
            self.gmem_params.items(),
            self.smem_params.items(),
            self.reg_params.items(),
        ))}\n)"
        return func_param_list

    def gen_func_arg_list(self, **kwargs) -> str:
        args = [kwargs[pn] for pn in chain(
            HW_IDX_PARAMS,
            self.sw_idx_params,
            self.const_params,
            self.gmem_params,
            self.smem_params,
            self.reg_params,
        )]
        return ', '.join(map(str, args))

    def gen_body(self, shifts: List[int]) -> str:
        body = self.body
        for pn, pt in HW_IDX_PARAMS.items():
            body = body.replace(f"{{{{HID.{pn}}}}}", pn)
        for pn, pt in self.sw_idx_params.items():
            body = body.replace(f"{{{{SID.{pn}}}}}", pn)
        for pn, pt in self.const_params.items():
            body = body.replace(f"{{{{C.{pn}}}}}", pn)
        for pn, pt in self.gmem_params.items():
            body = body.replace(f"{{{{G.{pn}}}}}", pn)
        for pn, pt in self.smem_params.items():
            body = body.replace(f"{{{{S.{pn}}}}}", pn)
        for pn, pt in self.reg_params.items():
            body = body.replace(f"{{{{R.{pn}}}}}", pn)
        for i in range(len(shifts) + 1):
            body = body.replace(f"{{{{RID.{i}}}}}", f"_v_r{i}")
        for i in range(len(shifts)):
            body = body.replace(f"{{{{SFT.{i}}}}}", f"_v_shift{i}")
        body = body.replace(f"{{{{WTN}}}}", f"wait_num")
        return body


@dataclass
class Op:
    name: str
    type: str
    src_template: SourceTemplate

    @staticmethod
    def gen_template_param_list(shifts: List[int]):
        return f"template <{', '.join(pt + ' ' + pn for pn, pt in chain(
            [(f'_v_shift{i}', "int",) for i in range(len(shifts))],
            [("max_shift", "int",)],
            [(f'_v_r{i}', "int",) for i in range(len(shifts) + 1)],
        ))}>"

    @staticmethod
    def gen_template_arg_list(shifts: List[int], reg_ids: List[int]):
        return f"<{', '.join(map(str, chain(
            shifts, [max(shifts) + 1], reg_ids
        )))}>"

    def gen_func(self, shifts: List[int]) -> str:
        template_param_list_src = self.gen_template_param_list(shifts)
        body_src = self.src_template.gen_body(shifts)
        func_param_list_src = self.src_template.gen_func_param_list()

        func_src = ("{{template_param_list}}\n"
                    "__device__ void {{name}}{{func_param_list}} {\n"
                    "{{body}}\n"
                    "}}")

        func_src = func_src.replace("{{name}}", self.name)
        func_src = func_src.replace("{{template_param_list}}", template_param_list_src)
        func_src = func_src.replace("{{func_param_list}}", func_param_list_src)
        func_src = func_src.replace("{{body}}", body_src.strip("\n"))

        return func_src

    def gen_func_call(self, shifts: List[int], reg_ids, **kwargs) -> str:
        template_arg_list_src = self.gen_template_arg_list(shifts, reg_ids)
        func_arg_list_src = self.src_template.gen_func_arg_list(**kwargs)

        func_call_src = ("{{name}}{{template_arg_list}}(\n"
                         "    {{func_arg_list}}\n"
                         ");")

        func_call_src = func_call_src.replace("{{name}}", self.name)
        func_call_src = func_call_src.replace("{{template_arg_list}}", template_arg_list_src)
        func_call_src = func_call_src.replace("{{func_arg_list}}", func_arg_list_src)

        return func_call_src


@dataclass
class MetaOp(Op):
    graph: nx.Graph
    ops: OrderedDict[str, Op]


@dataclass
class SequentialOp(Op):
    ops: List[Op]


@dataclass
class Stage:
    name: str
    ops: List[SequentialOp]


@dataclass
class Pipeline:
    stages: List[Stage]
    shifts: List[int]

    def gen_kernel(self):
        # TODO
        raise NotImplementedError()

    def gen_short_pipe(self):
        # TODO
        raise NotImplementedError()

    def gen_fill_stage(self, stage_id):
        # TODO
        raise NotImplementedError()

    def gen_fill(self):
        # TODO
        raise NotImplementedError()

    def gen_empty_stage_impl(self, stage_id):
        # TODO
        raise NotImplementedError()

    def gen_empty_stage(self, stage_id):
        # TODO
        raise NotImplementedError()

    def gen_empty(self):
        # TODO
        raise NotImplementedError()

    def gen_main_loop(self):
        # TODO
        raise NotImplementedError()

    def gen_main_loop_unrolled_step(self):
        # TODO
        raise NotImplementedError()

    def gen_main_loop_remainder(self):
        # TODO
        raise NotImplementedError()

    def gen_main_loop_remainder_impl(self):
        # TODO
        raise NotImplementedError()
