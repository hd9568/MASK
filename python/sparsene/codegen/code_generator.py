from format.format import Format


# The base class of all generators
class CodeGenerator:
    def __init__(self) -> None:
        pass

    def generate(self, format: Format):
        pass

    def generate_graph(self, format: Format):
        pass

    def emit_source(self, format: Format):
        pass
