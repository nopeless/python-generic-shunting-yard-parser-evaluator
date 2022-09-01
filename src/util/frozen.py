class FrozenList(list):
    def _immutable(self, *args, **kws):
        raise TypeError("cannot change object - object is immutable")

    pop = _immutable
    remove = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    reverse = _immutable


class FrozenDict(dict):
    def _immutable(self, *args, **kws):
        raise TypeError("cannot change object - object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
