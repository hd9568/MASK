from itertools import chain
from typing import List
from dataclasses import dataclass, field

from .op import NUM_RES_GROUPS_VAR_NAME, SequentialOp
from .param import Params


@dataclass(kw_only=True)
class Stage:
    name: str
    seq: SequentialOp

    def gen_kernel(self, num_stages: int) -> str:
        return self.seq.gen_func_def(num_stages)

    def gen_all_kernels(self, num_stages: int) -> str:
        op_kernels = map(lambda op: op.gen_func_def(num_stages), self.seq.ops)
        return "\n".join(chain(op_kernels, [self.seq.gen_func_def(num_stages)]))


@dataclass(kw_only=True)
class Pipeline:
    name: str
    stages: List[Stage]
    shifts: List[int]
    params: Params = field(init=False)

    def __post_init__(self):
        params = Params(is_global=True)
        for stage in self.stages:
            opp = stage.seq.get_params()
            params.const.update(opp.const)
            params.gmem.update(opp.gmem)

        self.params = params

    def max_shift(self) -> int:
        return max(self.shifts)

    def num_res_groups(self) -> int:
        return self.max_shift() + 1

    def num_stages(self) -> int:
        return len(self.stages)

    def gen_global_param_list(self):
        return self.params.gen_param_list()

    def gen_global_arg_list(self, **kwargs):
        return self.params.gen_arg_list(**kwargs)

    def gen_global_tparam_list(self):
        return f"template <{', '.join(pt + ' ' + pn for pn, pt in chain(
            [(f'_v_shift{i}', "int",) for i in range(self.num_stages())],
            [(NUM_RES_GROUPS_VAR_NAME, "int",)],
        ))}>"

    def gen_global_targ_list(self):
        return f"<{', '.join(map(str, chain(self.shifts, [self.num_res_groups()])))}>"

    def gen_global_func_def(self):
        kernel_src = (
            "{{tparam_list}}\n"
            "__global__ void {{name}}{{param_list}} {\n"
            "{{body}}\n"
            "}"
        )

        tparam_list_src = self.gen_global_tparam_list()
        param_list_src = self.gen_global_param_list()
        body = self.gen_body()

        kernel_src = kernel_src.replace("{{name}}", self.name)
        kernel_src = kernel_src.replace("{{tparam_list}}", tparam_list_src)
        kernel_src = kernel_src.replace("{{param_list}}", param_list_src)
        kernel_src = kernel_src.replace("{{body}}", body)

        return kernel_src

    def gen_kernel(self) -> str:
        return self.gen_global_func_def()

    def gen_all_kernels(self) -> str:
        stage_kernels = map(
            lambda stage: stage.gen_all_kernels(len(self.stages)), self.stages
        )
        return "\n".join(chain(stage_kernels, [self.gen_kernel()]))

    def gen_short_pipe_impl(self, num_steps: int) -> str:
        # TODO
        raise NotImplementedError()

    def gen_short_pipe(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_fill_stage(self, stage_id) -> str:
        # TODO
        raise NotImplementedError()

    def gen_fill(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_empty_stage_impl(self, stage_id) -> str:
        # TODO
        raise NotImplementedError()

    def gen_empty_stage(self, stage_id) -> str:
        # TODO
        raise NotImplementedError()

    def gen_empty(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_main_loop(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_main_loop_unrolled_step(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_main_loop_remainder(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_main_loop_remainder_impl(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_smem_allocation(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_reg_allocation(self) -> str:
        # TODO
        raise NotImplementedError()

    def gen_hid_init(self) -> str:
        hid_init_srcs = []
        for pn, pt in self.params.HW_IDX.items():
            hid_init_srcs.append("")

    def gen_body(self) -> str:
        # hid init
        # const init
        # smem def
        # reg def
        # TODO
        raise NotImplementedError()
