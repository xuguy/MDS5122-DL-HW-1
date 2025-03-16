is_simple_core = False

# from now on, there will be no need to use core_simple anymore
# but we want to keep consistent with the book so i keep this if condition
if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import Parameter
    from dezero.core import test_mode
    from dezero.core import Config
    from dezero.layers import Layer
    from dezero.models import Model


    # from dezero.dataloaders import Dataset
    from dezero.dataloaders import DataLoader
    # visit dataloader: from dezero import DataLoader

    import dezero.datasets
    import dezero.dataloaders
    import dezero.optimizers
    import dezero.layers
    import dezero.functions
    import dezero.functions_conv
    
    import dezero.utils
    import dezero.cuda
    # import dezero.transforms


# 为什么用dezero.core_simple？这个__init__文件不是已经和core_simple.py在同一个目录下吗？
# 因为我们总是从外部调用这个文件，我们会首先把dezero这个文件的父目录加入到全局变量中，这样我们就可以import dezero这个包。当我们import dezero后，__init__.py会被首先自动调用，但是这个时候，全局变量中保存的是dezero的父目录，因此__init__.py的import并不能识别到dezero内部的结构，因此需要从dezero的父目录开始寻找，所以需要dezero.core_simple，告诉import语句应该进入dezero包内的core_simple中import相应的模块

# 当我们从外部import dezero时，__init__.py里面加载的module（class/function)会被加载到dezero所属的name space中，如果要使用，必须通过dezero.Variable/dezero.test_mode来调用，除非使用from ... import ...把dezero的name space中的模块导入到现有的name space中

setup_variable()