from frozen import FrozenDict, FrozenList


class LexicalScope:
    def __init__(self, parent=None, identifiers=None):
        if not isinstance(parent, LexicalScope):
            if isinstance(parent, dict) and identifiers is None:
                identifiers = parent
                parent = None
        self.parent = parent
        self._identifiers = identifiers or {}

    def get_identifier_scope(self, name):
        # Prevent arguments from bleeding to inner scope
        # Not a top down solution but it exists for now
        if name.startswith("$"):
            return self
        if name in self._identifiers:
            return self
        if self.parent:
            return self.parent.get_identifier_scope(name)
        return None

    def get_identifier(self, name):
        if i := self.get_identifier_scope(name):
            return i._identifiers[name]
        return None

    def init_identifier(self, name, value=None):
        """
        Doesn't reuse outerscope
        """
        self._identifiers[name] = value

    def set_identifier(self, name, value):
        """
        Reuse outerscope
        """
        if i := self.get_identifier_scope(name):
            i._identifiers[name] = value
        else:
            self._identifiers[name] = value

    def inner_scope(self, identifiers):
        return LexicalScope(self, identifiers)


class NullScope(LexicalScope):
    def __init__(self):
        super().__init__(None, FrozenDict())
