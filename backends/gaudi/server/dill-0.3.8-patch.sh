#!/bin/bash
git clone -b 0.3.8 https://github.com/uqfoundation/dill.git
pushd dill
cat <<EOF > dill-0.3.8.patch
diff --git a/dill/_dill.py b/dill/_dill.py
index d42432f..1d251e6 100644
--- a/dill/_dill.py
+++ b/dill/_dill.py
@@ -69,7 +69,15 @@ TypeType = type # 'new-style' classes #XXX: unregistered
 XRangeType = range
 from types import MappingProxyType as DictProxyType, new_class
 from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, PickleError, PicklingError, UnpicklingError
-import __main__ as _main_module
+class _LazyMainModule(object):
+    _module = None
+    @property
+    def module(self):
+        if self._module is None:
+            import __main__ as _m_module
+            self._module = _m_module
+        return self._module
+_main_module = _LazyMainModule()
 import marshal
 import gc
 # import zlib
@@ -355,7 +363,7 @@ class Pickler(StockPickler):
         _fmode = kwds.pop('fmode', None)
         _recurse = kwds.pop('recurse', None)
         StockPickler.__init__(self, file, *args, **kwds)
-        self._main = _main_module
+        self._main = _main_module.module
         self._diff_cache = {}
         self._byref = settings['byref'] if _byref is None else _byref
         self._strictio = False #_strictio
@@ -437,12 +445,12 @@ class Unpickler(StockUnpickler):
         settings = Pickler.settings
         _ignore = kwds.pop('ignore', None)
         StockUnpickler.__init__(self, *args, **kwds)
-        self._main = _main_module
+        self._main = _main_module.module
         self._ignore = settings['ignore'] if _ignore is None else _ignore
 
     def load(self): #NOTE: if settings change, need to update attributes
         obj = StockUnpickler.load(self)
-        if type(obj).__module__ == getattr(_main_module, '__name__', '__main__'):
+        if type(obj).__module__ == getattr(self._main, '__name__', '__main__'):
             if not self._ignore:
                 # point obj class to main
                 try: obj.__class__ = getattr(self._main, type(obj).__name__)
@@ -1199,11 +1207,11 @@ def save_module_dict(pickler, obj):
         logger.trace(pickler, "D1: %s", _repr_dict(obj)) # obj
         pickler.write(bytes('c__builtin__\n__main__\n', 'UTF-8'))
         logger.trace(pickler, "# D1")
-    elif (not is_dill(pickler, child=False)) and (obj == _main_module.__dict__):
+    elif (not is_dill(pickler, child=False)) and (obj == _main_module.module.__dict__):
         logger.trace(pickler, "D3: %s", _repr_dict(obj)) # obj
         pickler.write(bytes('c__main__\n__dict__\n', 'UTF-8'))  #XXX: works in general?
         logger.trace(pickler, "# D3")
-    elif '__name__' in obj and obj != _main_module.__dict__ \\
+    elif '__name__' in obj and obj != _main_module.module.__dict__ \\
             and type(obj['__name__']) is str \\
             and obj is getattr(_import_module(obj['__name__'],True), '__dict__', None):
         logger.trace(pickler, "D4: %s", _repr_dict(obj)) # obj
diff --git a/dill/session.py b/dill/session.py
index e91068a..a921b43 100644
--- a/dill/session.py
+++ b/dill/session.py
@@ -233,7 +233,7 @@ def dump_module(
     protocol = settings['protocol']
     main = module
     if main is None:
-        main = _main_module
+        main = _main_module.module
     elif isinstance(main, str):
         main = _import_module(main)
     if not isinstance(main, ModuleType):
@@ -501,7 +501,7 @@ def load_module(
             pass
     assert loaded is main
     _restore_modules(unpickler, main)
-    if main is _main_module or main is module:
+    if main is _main_module.module or main is module:
         return None
     else:
         return main

EOF
git apply dill-0.3.8.patch
python -m pip install .
popd
rm -fr dill