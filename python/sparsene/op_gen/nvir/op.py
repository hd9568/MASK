from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain
from typing import List

import networkx as nx

from .param import Params
from sparsene.utils.source import indent

NUM_RES_GROUPS_VAR_NAME = "_v_num_res_groups"
WAIT_NUM_VAR_NAME = "_v_wait_num"


@dataclass(kw_only=True)
class OpBase(ABC):
    name: str
    type: str

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def gen_tparam_list(self, num_stages) -> str:
        pass

    @abstractmethod
    def gen_targ_list(
        self,
        shifts: List[int | str],
        num_res_groups: int | str,
        reg_ids: List[int | str],
        wait_num: int | str,
    ) -> str:
        pass

    @abstractmethod
    def gen_param_list(self) -> str:
        pass

    @abstractmethod
    def gen_arg_list(self, **kwargs) -> str:
        pass

    @abstractmethod
    def gen_func_def(self, num_stages: int) -> str:
        pass

    @abstractmethod
    def gen_func_call(
        self,
        shifts: List[int | str],
        num_res_groups: int | str,
        reg_ids: List[int | str],
        wait_num: int | str,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def gen_default_func_call(self, num_stages) -> str:
        pass


@dataclass(kw_only=True)
class Op(OpBase):
    func_body: str
    params: Params

    def get_params(self):
        return self.params

    def gen_tparam_list(self, num_stages: int) -> str:
        return f"template <{', '.join(pt + ' ' + pn for pn, pt in chain(
            [(f'_v_shift{i}', "int",) for i in range(num_stages)],
            [(NUM_RES_GROUPS_VAR_NAME, "int",)],
            [(f'_v_r{i}', "int",) for i in range(num_stages + 1)],
            [(WAIT_NUM_VAR_NAME, "int",)],
        ))}>"

    def gen_targ_list(
        self,
        shifts: List[int | str],
        num_res_groups: int | str,
        reg_ids: List[int | str],
        wait_num: int | str,
    ) -> str:
        return f"<{', '.join(map(str, chain(
            shifts, [num_res_groups], reg_ids, [wait_num]
        )))}>"

    def gen_param_list(self) -> str:
        return self.params.gen_param_list()

    def gen_arg_list(self, **kwargs) -> str:
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
        tparam_list_src = self.gen_tparam_list(num_stages)
        body_src = self.gen_func_body(num_stages)
        param_list_src = self.gen_param_list()

        func_src = (
            "{{tparam_list}}\n"
            "__device__ void {{name}}{{param_list}} {\n"
            "{{body}}\n"
            "}"
        )

        func_src = func_src.replace("{{name}}", self.name)
        func_src = func_src.replace("{{tparam_list}}", tparam_list_src)
        func_src = func_src.replace("{{param_list}}", param_list_src)
        func_src = func_src.replace("{{body}}", body_src.strip("\n"))

        return func_src

    def gen_func_call(
        self,
        shifts: List[int | str],
        num_res_groups: int | str,
        reg_ids: List[int | str],
        wait_num: int | str,
        **kwargs,
    ) -> str:
        targ_list_src = self.gen_targ_list(shifts, num_res_groups, reg_ids, wait_num)
        arg_list_src = self.gen_arg_list(**kwargs)

        func_call_src = "{{name}}{{targ_list}}(\n" "    {{arg_list}}\n" ");"

        func_call_src = func_call_src.replace("{{name}}", self.name)
        func_call_src = func_call_src.replace("{{targ_list}}", targ_list_src)
        func_call_src = func_call_src.replace("{{arg_list}}", arg_list_src)

        return func_call_src

    def gen_default_func_call(self, num_stages) -> str:
        return self.gen_func_call(
            shifts=[f"_v_shift{i}" for i in range(num_stages)],
            num_res_groups=NUM_RES_GROUPS_VAR_NAME,
            reg_ids=[f"_v_r{i}" for i in range(num_stages + 1)],
            wait_num=WAIT_NUM_VAR_NAME,
            **self.get_params().default_kwargs(),
        )


@dataclass(kw_only=True)
class MetaOp:
    graph: nx.Graph
    ops: OrderedDict[str, Op]


@dataclass(kw_only=True)
class SequentialOp(OpBase):
    ops: List[Op]
    params: Params = field(init=False)

    def __post_init__(self):
        params = Params()
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

    def gen_tparam_list(self, num_stages) -> str:
        return f"template <{', '.join(pt + ' ' + pn for pn, pt in chain(
            [(f'_v_shift{i}', "int",) for i in range(num_stages)],
            [(NUM_RES_GROUPS_VAR_NAME, "int",)],
            [(f'_v_r{i}', "int",) for i in range(num_stages + 1)],
            [(WAIT_NUM_VAR_NAME, "int",)],
        ))}>"

    def gen_targ_list(
        self,
        shifts: List[int | str],
        num_res_groups: int | str,
        reg_ids: List[int | str],
        wait_num: int | str,
    ) -> str:
        return f"<{', '.join(map(str, chain(
            shifts, [num_res_groups], reg_ids, [wait_num]
        )))}>"

    def gen_param_list(self) -> str:
        return self.params.gen_param_list()

    def gen_arg_list(self, **kwargs) -> str:
        return self.params.gen_arg_list(**kwargs)

    def gen_func_body(self, num_stages: int) -> str:
        return "\n".join(
            map(lambda op: indent(op.gen_default_func_call(num_stages)), self.ops)
        )

    def gen_func_def(self, num_stages: int) -> str:
        tparam_list_src = self.gen_tparam_list(num_stages)
        body_src = self.gen_func_body(num_stages)
        param_list_src = self.gen_param_list()

        func_src = (
            "{{tparam_list}}\n"
            "__device__ void {{name}}{{param_list}} {\n"
            "{{body}}\n"
            "}"
        )

        func_src = func_src.replace("{{name}}", self.name)
        func_src = func_src.replace("{{tparam_list}}", tparam_list_src)
        func_src = func_src.replace("{{param_list}}", param_list_src)
        func_src = func_src.replace("{{body}}", body_src.strip("\n"))
        return func_src

    def gen_func_call(
        self,
        shifts: List[int | str],
        num_res_groups: int | str,
        reg_ids: List[int | str],
        wait_num: int | str,
        **kwargs,
    ) -> str:
        targ_list_src = self.gen_targ_list(shifts, num_res_groups, reg_ids, wait_num)
        arg_list_src = self.gen_arg_list(**kwargs)

        func_call_src = "{{name}}{{targ_list}}(\n" "    {{arg_list}}\n" ");"

        func_call_src = func_call_src.replace("{{name}}", self.name)
        func_call_src = func_call_src.replace("{{targ_list}}", targ_list_src)
        func_call_src = func_call_src.replace("{{arg_list}}", arg_list_src)

        return func_call_src

    def gen_default_func_call(self, num_stages) -> str:
        return self.gen_func_call(
            shifts=[f"_v_shift{i}" for i in range(num_stages)],
            num_res_groups=NUM_RES_GROUPS_VAR_NAME,
            reg_ids=[f"_v_r{i}" for i in range(num_stages + 1)],
            wait_num=WAIT_NUM_VAR_NAME,
            **self.get_params().default_kwargs(),
        )
