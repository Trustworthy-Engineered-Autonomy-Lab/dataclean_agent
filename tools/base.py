class Tool:
    name: str = ""
    description: str = ""
    parameters: dict = {}
    _registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name") and cls.name:
            Tool._registry[cls.name] = cls

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise KeyError(f"Tool '{name}' is not registered in Tool._registry.")
        return cls._registry[name]()

    @classmethod
    def all_schemas(cls):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_cls.name,
                    "description": tool_cls.description,
                    "parameters": tool_cls.parameters
                }
            }
            for tool_cls in cls._registry.values()
            if tool_cls.name and getattr(tool_cls, "agent_exposed", True)
        ]

    def run(self, **kwargs):
        raise NotImplementedError("Tool run method must be implemented.")
