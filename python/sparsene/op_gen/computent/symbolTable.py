class SymbolTable:
    _instances: dict[str, "SymbolTable"] = {}

    def __init__(self):
        self._table = {}

    @classmethod
    def global_table(cls, name: str = "default") -> "SymbolTable":
        """
        获取指定名字的全局符号表，如果不存在则新建
        例如:
            SymbolTable.global_table("arraydef")
            SymbolTable.global_table("axis")
        """
        if name not in cls._instances:
            cls._instances[name] = cls()
        return cls._instances[name]

    @classmethod
    def reset(cls):
        """清空所有全局符号表"""
        cls._instances.clear()

    def clear(self):
        """清空当前符号表"""
        self._table.clear()

    def add(self, name: str, value):
        """添加一个符号，如果已存在则覆盖"""
        self._table[name] = value

    def get(self, name: str):
        """查找符号，存在返回 value，不存在返回 None"""
        return self._table.get(name, None)

    def exists(self, name: str) -> bool:
        """检查符号是否存在"""
        return name in self._table

    def remove(self, name: str):
        """删除符号，不存在则忽略"""
        self._table.pop(name, None)

    def __getitem__(self, name: str):
        return self._table[name]

    def __setitem__(self, name: str, value):
        self._table[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._table

    def __str__(self):
        return str(self._table)