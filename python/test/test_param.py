from sparsene.op_gen.nvir.param import Param

if __name__ == "__main__":
    params = [Param("a", "int"), Param("b", "float *"), Param("c", "double[5][6][7]")]
    for p in params:
        print(p.gen_decl())
