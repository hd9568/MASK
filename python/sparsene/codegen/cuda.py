from code_generator import CodeGenerator
from format.format import Format


# The base class of all generators
class CudaCodeGenerator(CodeGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, format: Format):
        pass

    def generate_graph(self, format: Format):
        pass

    def emit_source(self, format: Format):
        pass
