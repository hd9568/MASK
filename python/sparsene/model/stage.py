from enum import Enum
from itertools import chain
from typing import List, Optional, Tuple, ClassVar
from dataclasses import dataclass, field
from collections import OrderedDict
from abc import ABC, abstractmethod
import networkx as nx
import re

NUM_RES_GROUPS_VAR_NAME = "_v_num_res_groups"
WAIT_NUM_VAR_NAME = "_v_wait_num"


def indent(s: str, width: int = 4):
    return re.sub(r'^', ' ' * width, s, flags=re.M)


@dataclass(kw_only=True)
class OpParams:
    HW_IDX: ClassVar[OrderedDict[str, str]] = OrderedDict([
        ("block_idx_x", "int"),
        ("lane_id", "int"),
    ])
    sw_idx: OrderedDict[str, str] = field(default_factory=OrderedDict)
    const: OrderedDict[str, str] = field(default_factory=OrderedDict)
    gmem: OrderedDict[str, str] = field(default_factory=OrderedDict)
    smem: OrderedDict[str, str] = field(default_factory=OrderedDict)
    reg: OrderedDict[str, str] = field(default_factory=OrderedDict)

    def gen_param_list(self) -> str:
        param_list = f"(\n{',\n'.join('  ' + pt + ' ' + pn for pn, pt in chain(
            map(lambda p: ("_H_" + p[0], p[1]), self.HW_IDX.items()),
            map(lambda p: ("_I_" + p[0], p[1]), self.sw_idx.items()),
            map(lambda p: ("_C_" + p[0], p[1]), self.const.items()),
            map(lambda p: ("_G_" + p[0], p[1]), self.gmem.items()),
            map(lambda p: ("_S_" + p[0], p[1]), self.smem.items()),
            map(lambda p: ("_R_" + p[0], p[1]), self.reg.items()),
        ))}\n)"
        return param_list

    def gen_arg_list(self, **kwargs) -> str:
        args = [kwargs[pn] for pn in chain(
            self.HW_IDX,
            self.sw_idx,
            self.const,
            self.gmem,
            self.smem,
            self.reg,
        )]
        return ', '.join(map(str, args))

    def default_kwargs(self):
        return {
            **{pn: "_H_" + pn for pn in self.HW_IDX},
            **{pn: "_I_" + pn for pn in self.sw_idx},
            **{pn: "_C_" + pn for pn in self.const},
            **{pn: "_G_" + pn for pn in self.gmem},
            **{pn: "_S_" + pn for pn in self.smem},
            **{pn: "_R_" + pn for pn in self.reg},
        }

    def gen_default_arg_list(self) -> str:
        return self.gen_arg_list(**self.default_kwargs())


@dataclass(kw_only=True)
class OpBase(ABC):
    name: str
    type: str

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def gen_template_param_list(self, num_stages) -> str:
        pass

    @abstractmethod
    def gen_template_arg_list(self, shifts: List[int | str], num_res_groups: int | str, reg_ids: List[int | str],
                              wait_num: int | str) -> str:
        pass

    @abstractmethod
    def gen_func_param_list(self) -> str:
        pass

    @abstractmethod
    def gen_func_arg_list(self, **kwargs) -> str:
        pass

    @abstractmethod
    def gen_func_def(self, num_stages: int) -> str:
        pass

    @abstractmethod
    def gen_func_call(self, shifts: List[int | str], num_res_groups: int | str, reg_ids: List[int | str],
                      wait_num: int | str, **kwargs) -> str:
        pass

    @abstractmethod
    def gen_default_func_call(self, num_stages) -> str:
        pass


@dataclass(kw_only=True)
class Op(OpBase):
    func_body: str
    params: OpParams

    def get_params(self):
        return self.params

    def gen_template_param_list(self, num_stages: int) -> str:
        return f"template <{', '.join(pt + ' ' + pn for pn, pt in chain(
            [(f'_v_shift{i}', "int",) for i in range(num_stages)],
            [(NUM_RES_GROUPS_VAR_NAME, "int",)],
            [(f'_v_r{i}', "int",) for i in range(num_stages + 1)],
            [(WAIT_NUM_VAR_NAME, "int",)],
        ))}>"

    def gen_template_arg_list(self, shifts: List[int | str], num_res_groups: int | str, reg_ids: List[int | str],
                              wait_num: int | str) -> str:
        return f"<{', '.join(map(str, chain(
            shifts, [num_res_groups], reg_ids, [wait_num]
        )))}>"

    def gen_func_param_list(self) -> str:
        return self.params.gen_param_list()

    def gen_func_arg_list(self, **kwargs) -> str:
        return self.params.gen_arg_list(**kwargs)

    def gen_func_body(self, num_stages: int) -> str:
        body = self.func_body
        for pn, pt in self.params.HW_IDX.items():
            body = body.replace(f"{{{{H.{pn}}}}}", "_H_" + pn)
        for pn, pt in self.params.sw_idx.items():
            body = body.replace(f"{{{{I.{pn}}}}}", "_I_" + pn)
        for pn, pt in self.params.const.items():
            body = body.replace(f"{{{{C.{pn}}}}}", "_C_" + pn)
        for pn, pt in self.params.gmem.items():
            body = body.replace(f"{{{{G.{pn}}}}}", "_G_" + pn)
        for pn, pt in self.params.smem.items():
            body = body.replace(f"{{{{S.{pn}}}}}", "_S_" + pn)
        for pn, pt in self.params.reg.items():
            body = body.replace(f"{{{{R.{pn}}}}}", "_R_" + pn)
        for i in range(num_stages + 1):
            body = body.replace(f"{{{{RID.{i}}}}}", f"_v_r{i}")
        for i in range(num_stages):
            body = body.replace(f"{{{{SFT.{i}}}}}", f"_v_shift{i}")
        body = body.replace(f"{{{{WTN}}}}", WAIT_NUM_VAR_NAME)
        return body

    def gen_func_def(self, num_stages: int) -> str:
        template_param_list_src = self.gen_template_param_list(num_stages)
        body_src = self.gen_func_body(num_stages)
        func_param_list_src = self.gen_func_param_list()

        func_src = ("{{template_param_list}}\n"
                    "__device__ void {{name}}{{func_param_list}} {\n"
                    "{{body}}\n"
                    "}")

        func_src = func_src.replace("{{name}}", self.name)
        func_src = func_src.replace("{{template_param_list}}", template_param_list_src)
        func_src = func_src.replace("{{func_param_list}}", func_param_list_src)
        func_src = func_src.replace("{{body}}", body_src.strip("\n"))

        return func_src

    def gen_func_call(self, shifts: List[int | str], num_res_groups: int | str, reg_ids: List[int | str],
                      wait_num: int | str, **kwargs) -> str:
        template_arg_list_src = self.gen_template_arg_list(shifts, num_res_groups, reg_ids, wait_num)
        func_arg_list_src = self.gen_func_arg_list(**kwargs)

        func_call_src = ("{{name}}{{template_arg_list}}(\n"
                         "    {{func_arg_list}}\n"
                         ");")

        func_call_src = func_call_src.replace("{{name}}", self.name)
        func_call_src = func_call_src.replace("{{template_arg_list}}", template_arg_list_src)
        func_call_src = func_call_src.replace("{{func_arg_list}}", func_arg_list_src)

        return func_call_src

    def gen_default_func_call(self, num_stages) -> str:
        return self.gen_func_call(
            shifts=[f'_v_shift{i}' for i in range(num_stages)],
            num_res_groups=NUM_RES_GROUPS_VAR_NAME,
            reg_ids=[f'_v_r{i}' for i in range(num_stages + 1)],
            wait_num=WAIT_NUM_VAR_NAME,
            **self.get_params().default_kwargs()
        )


@dataclass(kw_only=True)
class MetaOp:
    graph: nx.Graph
    ops: OrderedDict[str, Op]


@dataclass(kw_only=True)
class SequentialOp(OpBase):
    ops: List[Op]
    params: OpParams = field(init=False)

    def __post_init__(self):
        params = OpParams()
        for op in self.ops:
            opp = op.get_params()
            params.sw_idx.update(opp.sw_idx)
            params.const.update(opp.const)
            params.gmem.update(opp.gmem)
            params.smem.update(opp.smem)
            params.reg.update(opp.reg)

        self.params = params

    def get_params(self):
        return self.params

    def gen_template_param_list(self, num_stages) -> str:
        return f"template <{', '.join(pt + ' ' + pn for pn, pt in chain(
            [(f'_v_shift{i}', "int",) for i in range(num_stages)],
            [(NUM_RES_GROUPS_VAR_NAME, "int",)],
            [(f'_v_r{i}', "int",) for i in range(num_stages + 1)],
            [(WAIT_NUM_VAR_NAME, "int",)],
        ))}>"

    def gen_template_arg_list(self, shifts: List[int | str], num_res_groups: int | str, reg_ids: List[int | str],
                              wait_num: int | str) -> str:
        return f"<{', '.join(map(str, chain(
            shifts, [num_res_groups], reg_ids, [wait_num]
        )))}>"

    def gen_func_param_list(self) -> str:
        return self.params.gen_param_list()

    def gen_func_arg_list(self, **kwargs) -> str:
        return self.params.gen_arg_list(**kwargs)

    def gen_func_body(self, num_stages: int) -> str:
        return '\n'.join(map(lambda op: indent(op.gen_default_func_call(num_stages)), self.ops))

    def gen_func_def(self, num_stages: int) -> str:
        template_param_list_src = self.gen_template_param_list(num_stages)
        body_src = self.gen_func_body(num_stages)
        func_param_list_src = self.gen_func_param_list()

        func_src = ("{{template_param_list}}\n"
                    "__device__ void {{name}}{{func_param_list}} {\n"
                    "{{body}}\n"
                    "}")

        func_src = func_src.replace("{{name}}", self.name)
        func_src = func_src.replace("{{template_param_list}}", template_param_list_src)
        func_src = func_src.replace("{{func_param_list}}", func_param_list_src)
        func_src = func_src.replace("{{body}}", body_src.strip("\n"))
        return func_src

    def gen_func_call(self, shifts: List[int | str], num_res_groups: int | str, reg_ids: List[int | str],
                      wait_num: int | str, **kwargs) -> str:
        template_arg_list_src = self.gen_template_arg_list(shifts, num_res_groups, reg_ids, wait_num)
        func_arg_list_src = self.gen_func_arg_list(**kwargs)

        func_call_src = ("{{name}}{{template_arg_list}}(\n"
                         "    {{func_arg_list}}\n"
                         ");")

        func_call_src = func_call_src.replace("{{name}}", self.name)
        func_call_src = func_call_src.replace("{{template_arg_list}}", template_arg_list_src)
        func_call_src = func_call_src.replace("{{func_arg_list}}", func_arg_list_src)

        return func_call_src

    def gen_default_func_call(self, num_stages) -> str:
        return self.gen_func_call(
            shifts=[f'_v_shift{i}' for i in range(num_stages)],
            num_res_groups=NUM_RES_GROUPS_VAR_NAME,
            reg_ids=[f'_v_r{i}' for i in range(num_stages + 1)],
            wait_num=WAIT_NUM_VAR_NAME,
            **self.get_params().default_kwargs()
        )


@dataclass(kw_only=True)
class Stage:
    name: str
    seq: SequentialOp

    def gen_kernel(self, num_stages: int) -> str:
        func_srcs = chain(map(lambda op: op.gen_func_def(num_stages), self.seq.ops),
                          [self.seq.gen_func_def(num_stages)])
        return '\n'.join(func_srcs)


@dataclass(kw_only=True)
class Pipeline:
    stages: List[Stage]
    shifts: List[int]

    def num_res_groups(self) -> int:
        return max(self.shifts)

    def num_res_group(self) -> int:
        return self.num_res_groups() + 1

    def gen_kernel(self):
        # TODO
        raise NotImplementedError()

    def gen_short_pipe_impl(self, num_steps: int):
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
