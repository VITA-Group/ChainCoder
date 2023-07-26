"""
💡 💡 This file implements the helper class to parse python code into syntax tree.

Due to tree traversal algorithm, all classes, beside original attributes, should additionally hold 5 attributes:

children, children_glue_strings, primary_string, content_string, closing_string
"""




class TerminalClass:
    def __init__(self, content_string) -> None:
        self.children = []
        self.children_glue_strings = []
        self.primary_string = ''
        self.content_string = str(content_string)
        self.closing_string = ''

    def __str__(self) -> str:
        return self.content_string
    def __repr__(self) -> str:
        return self.content_string



class Store(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Store()')
class Load(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Load()')


class LtE(TerminalClass):
    def __init__(self) -> None:
        super().__init__('LtE()')
class GtE(TerminalClass):
    def __init__(self) -> None:
        super().__init__('GtE()')
class Lt(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Lt()')
class Gt(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Gt()')
class Eq(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Eq()')
class Add(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Add()')
class Sub(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Sub()')
class USub(TerminalClass):
    def __init__(self) -> None:
        super().__init__('USub()')
class UAdd(TerminalClass):
    def __init__(self) -> None:
        super().__init__('UAdd()')
class And(TerminalClass):
    def __init__(self) -> None:
        super().__init__('And()')
class Or(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Or()')
class Not(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Not()')
class Is(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Is()')
class FloorDiv(TerminalClass):
    def __init__(self) -> None:
        super().__init__('FloorDiv()')
class Mult(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Mult()')
class Div(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Div()')




class NotEq(TerminalClass):
    def __init__(self) -> None:
        super().__init__('NotEq()')
class Mod(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Mod()')
class Invert(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Invert()')
class BitOr(TerminalClass):
    def __init__(self) -> None:
        super().__init__('BitOr()')
class Del(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Del()')
class In(TerminalClass):
    def __init__(self) -> None:
        super().__init__('In()')
class NotIn(TerminalClass):
    def __init__(self) -> None:
        super().__init__('NotIn()')
class BitAnd(TerminalClass):
    def __init__(self) -> None:
        super().__init__('BitAnd()')
class Pass(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Pass()')
class Pow(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Pow()')
class IsNot(TerminalClass):
    def __init__(self) -> None:
        super().__init__('IsNot()')
class RShift(TerminalClass):
    def __init__(self) -> None:
        super().__init__('RShift()')
class LShift(TerminalClass):
    def __init__(self) -> None:
        super().__init__('LShift()')
class BitXor(TerminalClass):
    def __init__(self) -> None:
        super().__init__('BitXor()')
class Break(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Break()')
class Continue(TerminalClass):
    def __init__(self) -> None:
        super().__init__('Continue()')

class MatMult(TerminalClass):
    def __init__(self) -> None:
        super().__init__('MatMult()')


class _____(TerminalClass):
    """
    this is the template for zero argument class
    """
    def __init__(self) -> None:
        super().__init__('_____()')






class LeafNode:
    def __init__(self, primary_string, content_string, closing_string, is_from_isolated_value=False) -> None:
        self.children = []
        self.children_glue_strings = []
        self.primary_string = str(primary_string)

        # quite tricky here: must ensure names have str mark, while other vars do not have str mark
        self.content_string = repr(content_string)

        self.closing_string = str(closing_string)
    def __str__(self) -> str:
        return self.content_string
    def __repr__(self) -> str:
        return self.content_string


class TupleBranchNode:
    def __init__(self, primary_string, children, children_glue_strings) -> None:
        self.primary_string = primary_string
        self.children = [ensure_synTree_obj(node) for node in children]
        self.children_glue_strings = children_glue_strings
        self.closing_string = ')'
        self.content_string = ''
    def __str__(self) -> str:
        return self.primary_string[:-1]
    def __repr__(self) -> str:
        return self.primary_string[:-1]

class ListBranchNode(TupleBranchNode):
    def __init__(self, primary_string, children) -> None:
        assert primary_string[-1]=='['
        children_glue_strings = [['','']]
        for i in range(len(children)-1):
            children_glue_strings.append([',', ''])
        super().__init__(primary_string, children, children_glue_strings)
        self.closing_string = ']'
    def __str__(self) -> str:
        return self.primary_string[:-1]
    def __repr__(self) -> str:
        return self.primary_string[:-1]


class PointBranchNode:
    def __init__(self, primary_string, child) -> None:
        self.primary_string = primary_string
        self.children = [ensure_synTree_obj(child)]
        self.children_glue_strings = [['', '']]
        self.closing_string = ''
        self.content_string = ''
    def __str__(self) -> str:
        return self.primary_string[:-1]
    def __repr__(self) -> str:
        return self.primary_string[:-1]



def ensure_synTree_obj(node):
    is_synTree_obj = False
    for _base_cls in [LeafNode, TupleBranchNode, PointBranchNode, ListBranchNode, TerminalClass]:
        if issubclass(type(node), _base_cls):
            is_synTree_obj = True
    if is_synTree_obj:
        return node
    else:
        return LeafNode('', node, '', is_from_isolated_value=True)






# 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 
# 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 
# 🔨 🔨 🔨 🔨 Syntax Classes below 🔨 🔨 🔨 🔨 🔨 
# 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 
# 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 🔨 





class Assign(TupleBranchNode):
    def __init__(self, targets, value, type_comment) -> None:
        # ---- as-is ----
        self.targets = targets
        self.value = value
        self.type_comment = type_comment
        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Assign(',
            children = [
                ListBranchNode('targets=[', self.targets), 
                PointBranchNode('value=', self.value)
                ],
            children_glue_strings =[
                ['', ''], [',', ',type_comment=None'],
            ])


class Name(LeafNode):
    def __init__(self, id, ctx) -> None:
        # ---- as-is ----
        # assert type(id) is str
        self.id = id
        self.ctx = ctx
        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Name(id=',
            content_string = self.id,
            closing_string = f',ctx={str(ctx)})'
            )




class Module(TupleBranchNode):
    def __init__(self, body, type_ignores):
        # ---- as-is ----
        self.body = body
        self.type_ignores = type_ignores

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Module(',
            children = [ListBranchNode('body=[', self.body)],
            children_glue_strings =[
                ['', ',type_ignores=[]']
            ])


class FunctionDef(TupleBranchNode):
    def __init__(self, name,args,body,decorator_list,returns,type_comment):
        # ---- as-is ----
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns
        self.type_comment = type_comment

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'FunctionDef(',
            children = [
                PointBranchNode('name=', self.name), 
                PointBranchNode('args=', self.args), 
                ListBranchNode('body=[', self.body)
                ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', ',decorator_list=[],returns=None,type_comment=None'],
            ])


class arguments(TupleBranchNode):
    def __init__(self,posonlyargs,args,vararg,kwonlyargs,kw_defaults,kwarg,defaults):
        # ---- as-is ----
        self.posonlyargs = posonlyargs
        self.args = args
        self.vararg = vararg
        self.kwonlyargs = kwonlyargs
        self.kw_defaults = kw_defaults
        self.kwarg = kwarg
        self.defaults = defaults

        # ---- tree-structure ----
        if kwarg and defaults:
            children = [
                ListBranchNode('args=[', self.args), 
                PointBranchNode('kwarg=', self.kwarg), 
                ListBranchNode('defaults=[', self.defaults)
                ]
            children_glue_strings = [
            ['posonlyargs=[],', ',vararg=None,kwonlyargs=[],kw_defaults=[]'], [',', ''], [',', '']
          ]
        elif kwarg and (not defaults):
            children = [
                ListBranchNode('args=[', self.args), 
                PointBranchNode('kwarg=', self.kwarg)
                ]
            children_glue_strings = [
            ['posonlyargs=[],', ',vararg=None,kwonlyargs=[],kw_defaults=[]'], [',', ',defaults=[]']
          ]
        elif (not kwarg) and (defaults):
            children = [
                ListBranchNode('args=[', self.args), 
                ListBranchNode('defaults=[', self.defaults)
                ]
            children_glue_strings = [
            ['posonlyargs=[],', ',vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None'], [',', '']
            ]
        else:
            children = [
                ListBranchNode('args=[', self.args)
                ]
            children_glue_strings = [
            ['posonlyargs=[],', ',vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]']
            ]

        super().__init__(
            primary_string = 'arguments(',
            children = children,
            children_glue_strings = children_glue_strings
            )
            



class Call(TupleBranchNode):
    def __init__(self, func,args,keywords) -> None:
        # ---- as-is ----
        self.func = func
        self.args = args
        self.keywords = keywords

        # ---- tree-structure ----
        if keywords:
            super().__init__(
                primary_string = 'Call(',
                children = [
                    PointBranchNode('func=', self.func),
                    ListBranchNode('args=[', self.args),
                    ListBranchNode('keywords=[', self.keywords)
                    ],
                children_glue_strings =[
                    ['', ''], [',', ''], [',', '']
                ])
        else:
            super().__init__(
                primary_string = 'Call(',
                children = [
                    PointBranchNode('func=', self.func),
                    ListBranchNode('args=[', self.args),
                    ],
                children_glue_strings =[
                    ['', ''], [',', ',keywords=[]']
                ])


class arg(TupleBranchNode):
    def __init__(self, arg, annotation, type_comment) -> None:
        # ---- as-is ----
        self.arg = arg
        self.annotation = annotation
        self.type_comment = type_comment

        # ---- tree-structure ----
        if self.annotation is None:
            super().__init__(
                primary_string = 'arg(',
                children = [PointBranchNode('arg=', self.arg)],
                children_glue_strings =[
                    ['', ',annotation=None,type_comment=None']
                ])
        else:
            super().__init__(
                primary_string = 'arg(',
                children = [
                    PointBranchNode('arg=', self.arg),
                    PointBranchNode('annotation=', self.annotation)
                    ],
                children_glue_strings =[
                    ['',''], [',', ',type_comment=None']
                ])


class If(TupleBranchNode):
    def __init__(self, test,body,orelse) -> None:
        # ---- as-is ----
        self.test = test
        self.body = body
        self.orelse = orelse

        # ---- tree-structure ----
        if self.orelse:
            super().__init__(
                primary_string = 'If(',
                children = [
                    PointBranchNode('test=', self.test), 
                    ListBranchNode('body=[', self.body), 
                    ListBranchNode('orelse=[', self.orelse)
                    ],
                children_glue_strings =[
                    ['', ''], [',', ''], [',', '']
                ])
        else:
            super().__init__(
                primary_string = 'If(',
                children = [
                    PointBranchNode('test=', self.test), 
                    ListBranchNode('body=[', self.body), 
                    ],
                children_glue_strings =[
                    ['', ''], [',', ',orelse=[]']
                ])


class Compare(TupleBranchNode):
    def __init__(self, left, ops, comparators) -> None:
        # ---- as-is ----
        self.left = left
        self.ops = ops
        self.comparators = comparators

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Compare(',
            children = [
                PointBranchNode('left=',self.left), 
                ListBranchNode('ops=[', self.ops), 
                ListBranchNode('comparators=[', self.comparators)
                ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', '']
            ])


class Return(TupleBranchNode):
    def __init__(self, value) -> None:
        # ---- as-is ----
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Return(',
            children = [PointBranchNode('value=', self.value)],
            children_glue_strings =[
                ['', '']
            ])


class Constant(TupleBranchNode):
    def __init__(self, value, kind) -> None:
        # ---- as-is ----
        self.value = value
        self.kind = kind

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Constant(',
            children = [PointBranchNode('value=', self.value)],
            children_glue_strings =[
                ['', ',kind=None']
            ])



class Expr(TupleBranchNode):
    def __init__(self, value) -> None:
        # ---- as-is ----
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Expr(',
            children = [PointBranchNode('value=', self.value)],
            children_glue_strings =[
                ['', '']
            ])
        # self.children[0].father_pointer = self


class FormattedValue(TupleBranchNode):
    def __init__(self, value,conversion,format_spec) -> None:
        # ---- as-is ----
        self.value = value
        self.conversion = conversion
        self.format_spec = format_spec

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'FormattedValue(',
            children = [
                PointBranchNode('value=', self.value),
                PointBranchNode('conversion=', self.conversion),
                ],
            children_glue_strings =[
                ['', ''], [',', ',format_spec=None']
            ])



class JoinedStr(TupleBranchNode):
    def __init__(self, values) -> None:
        # ---- as-is ----
        self.values = values

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'JoinedStr(',
            children = [ListBranchNode('values=[', self.values)],
            children_glue_strings =[
                ['', '']
            ])


class List(TupleBranchNode):
    def __init__(self, elts, ctx) -> None:
        # ---- as-is ----
        self.elts = elts
        self.ctx = ctx

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'List(',
            children = [ListBranchNode('elts=[', self.elts)],
            children_glue_strings =[
                ['', f',ctx={self.ctx}']
            ])


class alias(TupleBranchNode):
    def __init__(self, name, asname) -> None:
        # ---- as-is ----
        self.name = name
        self.asname = asname

        # ---- tree-structure ----
        if self.asname:
            super().__init__(
                primary_string = 'alias(',
                children = [
                    PointBranchNode('name=', self.name),
                    PointBranchNode('asname=', self.asname)
                    ],
                children_glue_strings =[
                    ['', ''], [',', '']
                ])
        else:
            super().__init__(
                primary_string = 'alias(',
                children = [
                    PointBranchNode('name=', self.name),
                    ],
                children_glue_strings =[
                    ['', ',asname=None']
                ])


class ImportFrom(TupleBranchNode):
    def __init__(self, module, names, level) -> None:
        # ---- as-is ----
        self.module = module
        self.names = names
        self.level = level

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'ImportFrom(',
            children = [
                PointBranchNode('module=', self.module),
                ListBranchNode('names=[', self.names),
            ],
            children_glue_strings =[
                ['', ''], [',', f',level={self.level}']
            ])


class For(TupleBranchNode):
    def __init__(self, target,iter,body,orelse,type_comment) -> None:
        # ---- as-is ----
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse
        self.type_comment = type_comment

        # ---- tree-structure ----
        if not self.orelse:
            super().__init__(
                primary_string = 'For(',
                children = [
                    PointBranchNode('target=', self.target),
                    PointBranchNode('iter=', self.iter),
                    ListBranchNode('body=[', self.body)                    
                ],
                children_glue_strings =[
                    ['', ''], [',', ''], [',', ',orelse=[],type_comment=None']
                ])
        else:
            super().__init__(
                primary_string = 'For(',
                children = [
                    PointBranchNode('target=', self.target),
                    PointBranchNode('iter=', self.iter),
                    ListBranchNode('body=[', self.body),
                    ListBranchNode('orelse=[', self.orelse)                    
                ],
                children_glue_strings =[
                    ['', ''], [',', ''], [',', ''], [',', ',type_comment=None']
                ])


class Attribute(TupleBranchNode):
    def __init__(self, value,attr,ctx) -> None:
        # ---- as-is ----
        self.value = value
        self.attr = attr
        self.ctx = ctx

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Attribute(',
            children = [
                PointBranchNode('value=', self.value),
                PointBranchNode('attr=', self.attr)
            ],
            children_glue_strings =[
                ['', ''], [',', f',ctx={self.ctx}']
            ])

class BinOp(TupleBranchNode):
    def __init__(self, left, op, right) -> None:
        # ---- as-is ----
        self.left = left
        self.op = op
        self.right = right

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'BinOp(',
            children = [
                PointBranchNode('left=', self.left),
                PointBranchNode('op=', self.op),
                PointBranchNode('right=', self.right)
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', '']
            ])



class Lambda(TupleBranchNode):
    def __init__(self, args, body) -> None:
        # ---- as-is ----
        self.args = args
        self.body = body

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Lambda(',
            children = [
                PointBranchNode('args=', self.args),
                PointBranchNode('body=', self.body)
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])



class While(TupleBranchNode):
    def __init__(self, test,body,orelse) -> None:
        # ---- as-is ----
        self.test = test
        self.body = body
        self.orelse = orelse

        # ---- tree-structure ----
        if not self.orelse:
            super().__init__(
                primary_string = 'While(',
                children = [
                    PointBranchNode('test=', self.test),
                    ListBranchNode('body=[', self.body),
                ],
                children_glue_strings =[
                    ['', ''], [',', ',orelse=[]']
                ])
        else:
            super().__init__(
                primary_string = 'While(',
                children = [
                    PointBranchNode('test=', self.test),
                    ListBranchNode('body=[', self.body),
                    ListBranchNode('orelse=[', self.orelse),
                ],
                children_glue_strings =[
                    ['', ''], [',', ''], [',', '']
                ])


class ListComp(TupleBranchNode):
    def __init__(self, elt, generators) -> None:
        # ---- as-is ----
        self.elt = elt
        self.generators = generators

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'ListComp(',
            children = [
                PointBranchNode('elt=', self.elt),
                ListBranchNode('generators=[', self.generators),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])


class comprehension(TupleBranchNode):
    def __init__(self, target, iter, ifs, is_async) -> None:
        # ---- as-is ----
        self.target = target
        self.iter = iter
        self.ifs = ifs
        self.is_async = is_async

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'comprehension(',
            children = [
                PointBranchNode('target=', self.target),
                PointBranchNode('iter=', self.iter),
                ListBranchNode('ifs=[', self.ifs),
                PointBranchNode('is_async=', self.is_async),
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', ''], [',', '']
            ])





class Subscript(TupleBranchNode):
    def __init__(self, value, slice, ctx) -> None:
        # ---- as-is ----
        self.value = value
        self.slice = slice
        self.ctx = ctx

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Subscript(',
            children = [
                PointBranchNode('value=', self.value),
                PointBranchNode('slice=', self.slice),
            ],
            children_glue_strings =[
                ['', ''], [',', f',ctx={self.ctx}']
            ])
            


class Tuple(TupleBranchNode):
    def __init__(self, elts,ctx) -> None:
        # ---- as-is ----
        self.elts = elts
        self.ctx = ctx

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Tuple(',
            children = [ListBranchNode('elts=[', self.elts)],
            children_glue_strings =[
                ['', f',ctx={self.ctx}']
            ])

class UnaryOp(TupleBranchNode):
    def __init__(self, op,operand) -> None:
        # ---- as-is ----
        self.op = op
        self.operand = operand

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'UnaryOp(',
            children = [
                PointBranchNode('op=', self.op),
                PointBranchNode('operand=', self.operand),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])


class ClassDef(TupleBranchNode):
    def __init__(self, name,bases,keywords,body,decorator_list) -> None:
        # ---- as-is ----
        self.name = name
        self.bases = bases
        self.keywords = keywords
        self.body = body
        self.decorator_list = decorator_list
        self.name = name

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'ClassDef(',
            children = [
                PointBranchNode('name=', self.name),
                ListBranchNode('bases=[', self.bases),
                ListBranchNode('keywords=[', self.keywords),
                ListBranchNode('body=[', self.body),
                ListBranchNode('decorator_list=[', self.decorator_list),
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', ''], [',', ''], [',', '']
            ])


class Index(TupleBranchNode):
    def __init__(self, value) -> None:
        # ---- as-is ----
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Index(',
            children = [
                PointBranchNode('value=', self.value),
            ],
            children_glue_strings =[
                ['', '']
            ])


class Slice(TupleBranchNode):
    def __init__(self, lower,upper,step ) -> None:
        # ---- as-is ----
        self.lower = lower
        self.upper = upper
        self.step = step

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Slice(',
            children = [
                PointBranchNode('lower=', self.lower),
                PointBranchNode('upper=', self.upper),
                PointBranchNode('step=', self.step),
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', '']
            ])


class AugAssign(TupleBranchNode):
    def __init__(self, target, op, value) -> None:
        # ---- as-is ----
        self.target = target
        self.op = op
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'AugAssign(',
            children = [
                PointBranchNode('target=', self.target),
                PointBranchNode('op=', self.op),
                PointBranchNode('value=', self.value),
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', '']
            ])




class ExtSlice(TupleBranchNode):
    def __init__(self, dims) -> None:
        # ---- as-is ----
        self.dims = dims

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'ExtSlice(',
            children = [
                ListBranchNode('dims=[', self.dims),
            ],
            children_glue_strings =[
                ['', '']
            ])



class Import(TupleBranchNode):
    def __init__(self, names) -> None:
        # ---- as-is ----
        self.names = names

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Import(',
            children = [
                ListBranchNode('names=[', self.names),
            ],
            children_glue_strings =[
                ['', '']
            ])


class Dict(TupleBranchNode):
    def __init__(self, keys,values) -> None:
        # ---- as-is ----
        self.keys = keys
        self.values = values

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Dict(',
            children = [
                ListBranchNode('keys=[', self.keys),
                ListBranchNode('values=[', self.values),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])



class keyword(TupleBranchNode):
    def __init__(self, arg, value) -> None:
        # ---- as-is ----
        self.arg = arg
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'keyword(',
            children = [
                PointBranchNode('arg=', self.arg),
                PointBranchNode('value=', self.value),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])



class Assert(TupleBranchNode):
    def __init__(self, test, msg) -> None:
        # ---- as-is ----
        self.test = test
        self.msg = msg

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Assert(',
            children = [
                PointBranchNode('test=', self.test),
                PointBranchNode('msg=', self.msg),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])



class BoolOp(TupleBranchNode):
    def __init__(self, op, values) -> None:
        # ---- as-is ----
        self.op = op
        self.values = values

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'BoolOp(',
            children = [
                PointBranchNode('op=', self.op),
                ListBranchNode('values=[', self.values),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])


class Raise(TupleBranchNode):
    def __init__(self, exc, cause) -> None:
        # ---- as-is ----
        self.exc = exc
        self.cause = cause

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Raise(',
            children = [
                PointBranchNode('exc=', self.exc),
                PointBranchNode('cause=', self.cause),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])

class Delete(TupleBranchNode):
    def __init__(self, targets) -> None:
        # ---- as-is ----
        self.targets = targets

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Delete(',
            children = [
                ListBranchNode('targets=[', self.targets),
            ],
            children_glue_strings =[
                ['', '']
            ])

class IfExp(TupleBranchNode):
    def __init__(self, test,body,orelse) -> None:
        # ---- as-is ----
        self.test = test
        self.body = body
        self.orelse = orelse

        # ---- tree-structure ----
        if self.orelse:
            super().__init__(
                primary_string = 'IfExp(',
                children = [
                    PointBranchNode('test=', self.test), 
                    PointBranchNode('body=', self.body), 
                    PointBranchNode('orelse=', self.orelse)
                    ],
                children_glue_strings =[
                    ['', ''], [',', ''], [',', '']
                ])
        else:
            super().__init__(
                primary_string = 'IfExp(',
                children = [
                    PointBranchNode('test=', self.test), 
                    PointBranchNode('body=', self.body), 
                    ],
                children_glue_strings =[
                    ['', ''], [',', ',orelse=None)']
                ])

class Try(TupleBranchNode):
    def __init__(self, body,handlers,orelse,finalbody) -> None:
        # ---- as-is ----
        self.body = body
        self.handlers = handlers
        self.orelse = orelse
        self.finalbody = finalbody

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Try(',
            children = [
                ListBranchNode('body=[', self.body), 
                ListBranchNode('handlers=[', self.handlers),
                ListBranchNode('orelse=[', self.orelse),
                ListBranchNode('finalbody=[', self.finalbody),
                ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', ''], [',', '']
            ])



class ExceptHandler(TupleBranchNode):
    def __init__(self, type,name,body) -> None:
        # ---- as-is ----
        self.type = type
        self.name = name
        self.body = body

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'ExceptHandler(',
            children = [
                PointBranchNode('type=', self.type),
                PointBranchNode('name=', self.name),
                ListBranchNode('body=[', self.body),
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', '']
            ])


class Starred(TupleBranchNode):
    def __init__(self, value,ctx) -> None:
        # ---- as-is ----
        self.value = value
        self.ctx = ctx

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Starred(',
            children = [
                PointBranchNode('value=', self.value),
                PointBranchNode('ctx=', self.ctx),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])

class GeneratorExp(TupleBranchNode):
    def __init__(self, elt,generators) -> None:
        # ---- as-is ----
        self.elt = elt
        self.generators = generators

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'GeneratorExp(',
            children = [
                PointBranchNode('elt=', self.elt),
                ListBranchNode('generators=[', self.generators),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])

class Global(TupleBranchNode):
    def __init__(self, names) -> None:
        # ---- as-is ----
        self.names = names

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Global(',
            children = [
                ListBranchNode('names=[', self.names),
            ],
            children_glue_strings =[
                ['', '']
            ])

class AnnAssign(TupleBranchNode):
    def __init__(self, target,annotation,value,simple) -> None:
        # ---- as-is ----
        self.target = target
        self.annotation = annotation
        self.value = value
        self.simple = simple

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'AnnAssign(',
            children = [
                PointBranchNode('target=', self.target),
                PointBranchNode('annotation=', self.annotation),
                PointBranchNode('value=', self.value),
                PointBranchNode('simple=', self.simple),
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', ''], [',', '']
            ])

class DictComp(TupleBranchNode):
    def __init__(self, key,value,generators) -> None:
        # ---- as-is ----
        self.key = key
        self.value = value
        self.generators = generators

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'DictComp(',
            children = [
                PointBranchNode('key=', self.key),
                PointBranchNode('value=', self.value),
                ListBranchNode('generators=[', self.generators),
            ],
            children_glue_strings =[
                ['', ''], [',', ''], [',', '']
            ])

class Set(TupleBranchNode):
    def __init__(self, elts) -> None:
        # ---- as-is ----
        self.elts = elts

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Set(',
            children = [
                ListBranchNode('elts=[', self.elts),
            ],
            children_glue_strings =[
                ['', '']
            ])

class Yield(TupleBranchNode):
    def __init__(self, value) -> None:
        # ---- as-is ----
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Yield(',
            children = [
                PointBranchNode('value=', self.value),
            ],
            children_glue_strings =[
                ['', '']
            ])

class SetComp(TupleBranchNode):
    def __init__(self, elt,generators) -> None:
        # ---- as-is ----
        self.elt = elt
        self.generators = generators

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'SetComp(',
            children = [
                PointBranchNode('elt=', self.elt),
                ListBranchNode('generators=[', self.generators),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])

class With(TupleBranchNode):
    def __init__(self, items,body,type_comment) -> None:
        # ---- as-is ----
        self.items = items
        self.body = body
        self.type_comment = type_comment

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'With(',
            children = [
                ListBranchNode('items=[', self.items),
                ListBranchNode('body=[', self.body),
            ],
            children_glue_strings =[
                ['', ''], [',', ',type_comment=None']
            ])

class withitem(TupleBranchNode):
    def __init__(self, context_expr,optional_vars) -> None:
        # ---- as-is ----
        self.context_expr = context_expr
        self.optional_vars = optional_vars

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'withitem(',
            children = [
                PointBranchNode('context_expr=', self.context_expr),
                PointBranchNode('optional_vars=', self.optional_vars),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])

class Nonlocal(TupleBranchNode):
    def __init__(self, names) -> None:
        # ---- as-is ----
        self.names = names

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'Nonlocal(',
            children = [
                ListBranchNode('names=[', self.names),
            ],
            children_glue_strings =[
                ['', '']
            ])


class YieldFrom(TupleBranchNode):
    def __init__(self, value) -> None:
        # ---- as-is ----
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'YieldFrom(',
            children = [
                PointBranchNode('value=', self.value),
            ],
            children_glue_strings =[
                ['', '']
            ])



class NamedExpr(TupleBranchNode):
    def __init__(self, target,value) -> None:
        # ---- as-is ----
        self.target = target
        self.value = value

        # ---- tree-structure ----
        super().__init__(
            primary_string = 'NamedExpr(',
            children = [
                PointBranchNode('target=', self.target),
                PointBranchNode('value=', self.value),
            ],
            children_glue_strings =[
                ['', ''], [',', '']
            ])
