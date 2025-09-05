import ast
import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Callable, Optional, Dict, Any, Tuple, List
import time
import warnings
import json
from dataclasses import dataclass
from datetime import datetime
import base64
import io
import streamlit.components.v1 as components

# Configuraciones
warnings.filterwarnings('ignore')
plt.style.use('default')

# Procesar datos del teclado matemático
query_params = st.experimental_get_query_params()
if 'keyboard_input' in query_params:
    try:
        keyboard_data = json.loads(query_params['keyboard_input'][0])
        target_func = keyboard_data['target']
        expression = keyboard_data['expression']
        
        if target_func == 'fx':
            st.session_state.fx_expression = expression
        elif target_func == 'dfx':
            st.session_state.dfx_expression = expression
        elif target_func == 'gx':
            st.session_state.gx_expression = expression
            
        st.rerun()
    except:
        pass

# HTML para MathJax
MATHJAX_CONFIG = """
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
  }
};
</script>
"""

def render_math(expression: str) -> str:
    """Convierte expresión Python a LaTeX para renderizado matemático."""
    if not expression:
        return "..."
    
    conversions = {
        '**': '^',
        'sqrt(': '\\sqrt{',
        'sin(': '\\sin(',
        'cos(': '\\cos(',
        'tan(': '\\tan(',
        'asin(': '\\arcsin(',
        'acos(': '\\arccos(',
        'atan(': '\\arctan(',
        'sinh(': '\\sinh(',
        'cosh(': '\\cosh(',
        'tanh(': '\\tanh(',
        'exp(': '\\exp(',
        'log(': '\\ln(',
        'log10(': '\\log_{10}(',
        'log2(': '\\log_{2}(',
        'pi': '\\pi',
        'e': 'e',
        '*': '\\cdot '
    }
    
    latex_expr = expression
    for py_syntax, latex_syntax in conversions.items():
        latex_expr = latex_expr.replace(py_syntax, latex_syntax)
    
    # Balancear llaves para sqrt
    if '\\sqrt{' in latex_expr:
        parts = latex_expr.split('\\sqrt{')
        if len(parts) > 1:
            for i in range(1, len(parts)):
                if ')' in parts[i]:
                    parts[i] = parts[i].replace(')', '}', 1)
        latex_expr = '\\sqrt{'.join(parts)
    
    return latex_expr

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
    
    class SymbolicDerivative:
        """Calculadora de derivadas simbólicas usando SymPy."""
        
        @staticmethod
        def calculate_derivative(expression: str) -> str:
            """Calcula la derivada simbólica de una expresión."""
            try:
                x = sp.Symbol('x')
                expr_str = expression.replace('**', '^').replace('^', '**')
                expr = sp.sympify(expr_str)
                derivative = sp.diff(expr, x)
                derivative_str = str(derivative)
                
                replacements = {
                    'Abs(': 'abs(',
                    'log(': 'log(',
                    'exp(': 'exp(',
                    'sqrt(': 'sqrt(',
                    'sin(': 'sin(',
                    'cos(': 'cos(',
                    'tan(': 'tan(',
                    'asin(': 'asin(',
                    'acos(': 'acos(',
                    'atan(': 'atan(',
                    'sinh(': 'sinh(',
                    'cosh(': 'cosh(',
                    'tanh(': 'tanh(',
                }
                
                for sympy_func, python_func in replacements.items():
                    derivative_str = derivative_str.replace(sympy_func, python_func)
                
                return derivative_str
                
            except Exception as e:
                raise ValueError(f"Error calculando derivada: {str(e)}")
                
except ImportError:
    SYMPY_AVAILABLE = False
    
    class SymbolicDerivative:
        @staticmethod
        def calculate_derivative(expression: str) -> str:
            raise ValueError("SymPy no está disponible. Instala con: pip install sympy")

@dataclass
class MethodResult:
    """Estructura para almacenar resultados de métodos."""
    name: str
    root: Optional[float]
    history: List[Tuple]
    status: str
    execution_time: float
    iterations: int
    final_error: float
    convergence_rate: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'root': self.root,
            'status': self.status,
            'execution_time': self.execution_time,
            'iterations': self.iterations,
            'final_error': self.final_error,
            'convergence_rate': self.convergence_rate
        }

class FunctionValidator:
    """Validador avanzado de funciones matemáticas."""
    
    @staticmethod
    def validate_expression(expr: str) -> Tuple[bool, str, Optional[Callable]]:
        """Valida una expresión matemática."""
        if not expr.strip():
            return False, "La expresión no puede estar vacía", None
            
        try:
            func = FunctionValidator._make_safe_func(expr)
            test_values = [0.1, 1.0, -1.0, 2.0]
            for val in test_values:
                try:
                    result = func(val)
                    if math.isnan(result) or math.isinf(result):
                        continue
                except:
                    continue
            return True, "Expresión válida", func
        except Exception as e:
            return False, f"Error: {str(e)}", None
    
    @staticmethod
    def _make_safe_func(expr: str) -> Callable[[float], float]:
        """Crea función segura con validación mejorada."""
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
        allowed_names.update({
            "abs": abs, 
            "pow": pow, 
            "min": min, 
            "max": max,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "asinh": math.asinh,
            "acosh": math.acosh,
            "atanh": math.atanh
        })
        
        try:
            expr_ast = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Error de sintaxis: {e}")
        
        for node in ast.walk(expr_ast):
            if isinstance(node, ast.Name):
                if node.id != 'x' and node.id not in allowed_names:
                    raise ValueError(f"Variable/función no permitida: {node.id}")
        
        code = compile(expr_ast, '<string>', 'eval')
        
        def safe_func(x: float) -> float:
            try:
                result = eval(code, {'__builtins__': {}}, {**allowed_names, 'x': x})
                if math.isnan(result):
                    raise ValueError(f"Resultado NaN en x={x}")
                return float(result)
            except ZeroDivisionError:
                raise ValueError(f"División por cero en x={x}")
            except (OverflowError, ValueError) as e:
                raise ValueError(f"Error numérico en x={x}: {e}")
        
        return safe_func

class NumericalMethods:
    """Implementación avanzada de métodos numéricos."""
    
    @staticmethod
    def newton_raphson_adaptive(f: Callable, x0: float, df: Optional[Callable] = None,
                               tol: float = 1e-8, max_iter: int = 50) -> MethodResult:
        """Newton-Raphson con paso adaptativo."""
        history = []
        x = x0
        start_time = time.perf_counter()
        
        for n in range(max_iter):
            try:
                fx = f(x)
                dfx = df(x) if df else NumericalMethods._numerical_derivative(f, x)
                
                if abs(dfx) < 1e-14:
                    end_time = time.perf_counter()
                    return MethodResult(
                        "Newton-Raphson", None, history, 
                        "Derivada muy pequeña", end_time - start_time,
                        len(history), float('inf'), "No converge"
                    )
                
                step = fx / dfx
                alpha = 1.0
                x_new = x - alpha * step
                
                try:
                    while abs(f(x_new)) > abs(fx) and alpha > 0.1:
                        alpha *= 0.5
                        x_new = x - alpha * step
                except:
                    pass
                
                abs_err = abs(x_new - x)
                rel_err = abs_err / abs(x_new) if x_new != 0 else float('inf')
                
                history.append((n, x, fx, dfx, abs_err, rel_err))
                
                if abs_err < tol:
                    end_time = time.perf_counter()
                    convergence_rate = NumericalMethods._estimate_convergence_rate(history)
                    return MethodResult(
                        "Newton-Raphson", x_new, history,
                        "Convergencia exitosa", end_time - start_time,
                        len(history), abs_err, convergence_rate
                    )
                
                x = x_new
                
            except Exception as e:
                end_time = time.perf_counter()
                return MethodResult(
                    "Newton-Raphson", None, history,
                    f"Error: {str(e)}", end_time - start_time,
                    len(history), float('inf'), "Error"
                )
        
        end_time = time.perf_counter()
        return MethodResult(
            "Newton-Raphson", None, history,
            "Máximo de iteraciones", end_time - start_time,
            len(history), float('inf'), "No converge"
        )
    
    @staticmethod
    def secant_method(f: Callable, x0: float, x1: float,
                     tol: float = 1e-8, max_iter: int = 50) -> MethodResult:
        """Método de la secante."""
        history = []
        start_time = time.perf_counter()
        
        for n in range(max_iter):
            try:
                f0, f1 = f(x0), f(x1)
                
                if abs(f1 - f0) < 1e-14:
                    end_time = time.perf_counter()
                    return MethodResult(
                        "Secante", None, history,
                        "Denominador muy pequeño", end_time - start_time,
                        len(history), float('inf'), "No converge"
                    )
                
                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                abs_err = abs(x2 - x1)
                rel_err = abs_err / abs(x2) if x2 != 0 else float('inf')
                
                history.append((n, x0, x1, f0, f1, abs_err, rel_err))
                
                if abs_err < tol:
                    end_time = time.perf_counter()
                    convergence_rate = NumericalMethods._estimate_convergence_rate(history)
                    return MethodResult(
                        "Secante", x2, history,
                        "Convergencia exitosa", end_time - start_time,
                        len(history), abs_err, convergence_rate
                    )
                
                x0, x1 = x1, x2
                
            except Exception as e:
                end_time = time.perf_counter()
                return MethodResult(
                    "Secante", None, history,
                    f"Error: {str(e)}", end_time - start_time,
                    len(history), float('inf'), "Error"
                )
        
        end_time = time.perf_counter()
        return MethodResult(
            "Secante", None, history,
            "Máximo de iteraciones", end_time - start_time,
            len(history), float('inf'), "No converge"
        )
    
    @staticmethod
    def bisection_method(f: Callable, a: float, b: float,
                        tol: float = 1e-8, max_iter: int = 50) -> MethodResult:
        """Método de bisección."""
        history = []
        start_time = time.perf_counter()
        
        try:
            fa, fb = f(a), f(b)
            if fa * fb > 0:
                end_time = time.perf_counter()
                return MethodResult(
                    "Bisección", None, history,
                    "No hay cambio de signo en [a,b]", end_time - start_time,
                    0, float('inf'), "No aplicable"
                )
        except Exception as e:
            end_time = time.perf_counter()
            return MethodResult(
                "Bisección", None, history,
                f"Error evaluando función: {str(e)}", end_time - start_time,
                0, float('inf'), "Error"
            )
        
        for n in range(max_iter):
            try:
                c = (a + b) / 2
                fc = f(c)
                abs_err = abs(b - a) / 2
                rel_err = abs_err / abs(c) if c != 0 else float('inf')
                
                history.append((n, a, b, c, fa, fb, fc, abs_err, rel_err))
                
                if abs_err < tol or abs(fc) < tol:
                    end_time = time.perf_counter()
                    return MethodResult(
                        "Bisección", c, history,
                        "Convergencia exitosa", end_time - start_time,
                        len(history), abs_err, "Lineal"
                    )
                
                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
                    
            except Exception as e:
                end_time = time.perf_counter()
                return MethodResult(
                    "Bisección", None, history,
                    f"Error: {str(e)}", end_time - start_time,
                    len(history), float('inf'), "Error"
                )
        
        end_time = time.perf_counter()
        return MethodResult(
            "Bisección", (a + b) / 2, history,
            "Máximo de iteraciones", end_time - start_time,
            len(history), abs(b - a) / 2, "Lineal"
        )
    
    @staticmethod
    def punto_fijo(g: Callable, x0: float, tol: float = 1e-8, max_iter: int = 50) -> MethodResult:
        """Método de punto fijo."""
        history = []
        x = x0
        start_time = time.perf_counter()
        
        for n in range(max_iter):
            try:
                x_next = g(x)
                abs_err = abs(x_next - x)
                rel_err = abs_err / abs(x_next) if x_next != 0 else float('inf')
                
                history.append((n, x, x_next, abs_err, rel_err))
                
                if abs_err < tol:
                    end_time = time.perf_counter()
                    convergence_rate = NumericalMethods._estimate_convergence_rate(history)
                    return MethodResult(
                        "Punto Fijo", x_next, history,
                        "Convergencia exitosa", end_time - start_time,
                        len(history), abs_err, convergence_rate
                    )
                
                if abs(x_next) > 1e10:
                    end_time = time.perf_counter()
                    return MethodResult(
                        "Punto Fijo", None, history,
                        "Divergencia detectada", end_time - start_time,
                        len(history), float('inf'), "Diverge"
                    )
                
                x = x_next
                
            except Exception as e:
                end_time = time.perf_counter()
                return MethodResult(
                    "Punto Fijo", None, history,
                    f"Error: {str(e)}", end_time - start_time,
                    len(history), float('inf'), "Error"
                )
        
        end_time = time.perf_counter()
        return MethodResult(
            "Punto Fijo", None, history,
            "Máximo de iteraciones", end_time - start_time,
            len(history), float('inf'), "No converge"
        )
    
    @staticmethod
    def punto_fijo_aitken(g: Callable, x0: float, tol: float = 1e-8, max_iter: int = 50) -> MethodResult:
        """Método de punto fijo con aceleración de Aitken."""
        history = []
        x = x0
        start_time = time.perf_counter()
        
        for n in range(max_iter):
            try:
                x1 = g(x)
                x2 = g(x1)
                
                denom = x2 - 2*x1 + x
                if abs(denom) > 1e-14:
                    x_acc = x2 - (x2 - x1)**2 / denom
                else:
                    x_acc = x2
                
                abs_err = abs(x_acc - x)
                rel_err = abs_err / abs(x_acc) if x_acc != 0 else float('inf')
                
                history.append((n, x, x_acc, abs_err, rel_err))
                
                if abs_err < tol:
                    end_time = time.perf_counter()
                    convergence_rate = "Super-lineal (Aitken)"
                    return MethodResult(
                        "Aitken", x_acc, history,
                        "Convergencia exitosa", end_time - start_time,
                        len(history), abs_err, convergence_rate
                    )
                
                if abs(x_acc) > 1e10:
                    end_time = time.perf_counter()
                    return MethodResult(
                        "Aitken", None, history,
                        "Divergencia detectada", end_time - start_time,
                        len(history), float('inf'), "Diverge"
                    )
                
                x = x_acc
                
            except Exception as e:
                end_time = time.perf_counter()
                return MethodResult(
                    "Aitken", None, history,
                    f"Error: {str(e)}", end_time - start_time,
                    len(history), float('inf'), "Error"
                )
        
        end_time = time.perf_counter()
        return MethodResult(
            "Aitken", None, history,
            "Máximo de iteraciones", end_time - start_time,
            len(history), float('inf'), "No converge"
        )
    
    @staticmethod
    def _numerical_derivative(f: Callable, x: float, h: float = 1e-8) -> float:
        """Derivada numérica con diferencias centrales mejorada."""
        try:
            return (f(x + h) - f(x - h)) / (2 * h)
        except:
            try:
                return (f(x + h) - f(x)) / h
            except:
                raise ValueError(f"No se puede calcular derivada en x={x}")
    
    @staticmethod
    def _estimate_convergence_rate(history: List) -> str:
        """Estima la tasa de convergencia basada en el historial."""
        if len(history) < 3:
            return "Insuficiente data"
        
        errors = [h[-2] for h in history[-3:] if h[-2] > 0]
        if len(errors) < 3:
            return "No determinable"
        
        try:
            ratio1 = errors[-1] / errors[-2]
            ratio2 = errors[-2] / errors[-3]
            if ratio1 < 0.1 and ratio2 < 0.1:
                return "Cuadrática"
            elif ratio1 < 0.7:
                return "Super-lineal"
            else:
                return "Lineal"
        except:
            return "No determinable"

class AdvancedVisualizations:
    """Visualizaciones avanzadas con Plotly."""
    
    @staticmethod
    def create_interactive_convergence_plot(results: List[MethodResult]) -> go.Figure:
        """Gráfico interactivo de convergencia."""
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, result in enumerate(results):
            if result.history:
                iterations = [h[0] for h in result.history]
                errors = [h[-2] for h in result.history if h[-2] > 0]
                
                if errors:
                    fig.add_trace(go.Scatter(
                        x=iterations[:len(errors)],
                        y=errors,
                        mode='lines+markers',
                        name=result.name,
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=8),
                        hovertemplate=f"<b>{result.name}</b><br>" +
                                    "Iteración: %{x}<br>" +
                                    "Error: %{y:.2e}<br>" +
                                    "<extra></extra>"
                    ))
        
        fig.update_layout(
            title="Convergencia de Métodos Numéricos",
            xaxis_title="Iteración",
            yaxis_title="Error Absoluto",
            yaxis_type="log",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_function_plot_with_roots(f: Callable, results: List[MethodResult], 
                                      x_range: Tuple[float, float] = None) -> go.Figure:
        """Gráfico interactivo de función con raíces encontradas."""
        if x_range is None:
            all_points = []
            for result in results:
                if result.history:
                    points = [h[1] for h in result.history if isinstance(h[1], (int, float))]
                    all_points.extend(points)
            
            if all_points:
                x_min, x_max = min(all_points) - 2, max(all_points) + 2
            else:
                x_min, x_max = -5, 5
        else:
            x_min, x_max = x_range
        
        x_vals = np.linspace(x_min, x_max, 1000)
        try:
            y_vals = [f(x) for x in x_vals]
        except:
            x_vals = np.linspace(-5, 5, 1000)
            y_vals = [f(x) for x in x_vals]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='f(x)',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, result in enumerate(results):
            if result.root is not None:
                try:
                    fig.add_trace(go.Scatter(
                        x=[result.root],
                        y=[f(result.root)],
                        mode='markers',
                        name=f'Raíz {result.name}',
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=15,
                            symbol='star',
                            line=dict(color='black', width=2)
                        ),
                        hovertemplate=f"<b>{result.name}</b><br>" +
                                    "x = %{x:.6f}<br>" +
                                    "f(x) = %{y:.2e}<br>" +
                                    "<extra></extra>"
                    ))
                except:
                    pass
        
        fig.update_layout(
            title="Función y Raíces Encontradas",
            xaxis_title="x",
            yaxis_title="f(x)",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_performance_comparison(results: List[MethodResult]) -> go.Figure:
        """Comparación de rendimiento de métodos."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Tiempo de Ejecución', 'Número de Iteraciones', 
                          'Error Final', 'Tasa de Convergencia'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        names = [r.name for r in results if r.root is not None]
        times = [r.execution_time for r in results if r.root is not None]
        iterations = [r.iterations for r in results if r.root is not None]
        errors = [r.final_error for r in results if r.root is not None]
        convergence_rates = [r.convergence_rate for r in results if r.root is not None]
        
        if names:
            fig.add_trace(go.Bar(x=names, y=times, name="Tiempo (s)", 
                               marker_color='lightblue'), row=1, col=1)
            
            fig.add_trace(go.Bar(x=names, y=iterations, name="Iteraciones",
                               marker_color='lightgreen'), row=1, col=2)
            
            fig.add_trace(go.Bar(x=names, y=errors, name="Error Final",
                               marker_color='salmon'), row=2, col=1)
            
            if convergence_rates:
                convergence_counts = pd.Series(convergence_rates).value_counts()
                fig.add_trace(go.Pie(
                    labels=convergence_counts.index,
                    values=convergence_counts.values,
                    name="Convergencia"
                ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, template="plotly_white")
        return fig

class MathRenderer:
    """Renderizador de expresiones matemáticas."""
    
    @staticmethod
    def render_expression(expression: str, label: str = "f(x)") -> None:
        """Renderiza una expresión matemática usando MathJax."""
        if not expression:
            st.write("*No hay expresión para mostrar*")
            return
        
        latex_expr = render_math(expression)
        
        math_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <div style="text-align: center; font-size: 24px; margin-bottom: 10px;">
                $${label} = {latex_expr}$$
            </div>
            <div style="text-align: center; font-size: 14px; opacity: 0.8;">
                Código Python: <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px;">{expression}</code>
            </div>
        </div>
        """
        
        components.html(
            MATHJAX_CONFIG + math_html,
            height=120
        )
    
    @staticmethod
    def render_method_formula(method_name: str) -> None:
        """Renderiza la fórmula de un método específico."""
        formulas = {
            "Newton-Raphson": r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}",
            "Secante": r"x_{n+1} = x_n - f(x_n) \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}",
            "Bisección": r"c = \frac{a + b}{2}, \quad \text{luego } f(a)f(c) < 0 \Rightarrow b=c \text{ sino } a=c",
            "Punto Fijo": r"x_{n+1} = g(x_n)",
            "Aitken": r"x_{acc} = x_2 - \frac{(x_2 - x_1)^2}{x_2 - 2x_1 + x_0}"
        }
        
        if method_name in formulas:
            formula_html = f"""
            <div style="
                background: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 15px;
                margin: 10px 0;
                border-radius: 0 8px 8px 0;
            ">
                <h4 style="margin-top: 0; color: #007bff;">Fórmula: {method_name}</h4>
                <div style="text-align: center; font-size: 18px;">
                    ${formulas[method_name]}$
                </div>
            </div>
            """
            
            components.html(
                MATHJAX_CONFIG + formula_html,
                height=120
            )

def create_math_keyboard(target_function: str = "fx") -> str:
    """Teclado matemático completo que se integra con Streamlit."""
    keyboard_id = f"keyboard_{target_function}"
    
    return f"""
    <div id="{keyboard_id}" style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <div style="margin-bottom: 10px;">
            <label style="font-weight: bold;">Expresión actual:</label>
            <div id="display_{target_function}" style="background: white; border: 1px solid #ced4da; border-radius: 4px; padding: 8px; font-family: monospace; min-height: 30px;"></div>
        </div>
        
        <!-- Tabs para diferentes categorías de funciones -->
        <div style="margin-bottom: 10px;">
            <button onclick="showTab_{target_function}('basic')" id="tab_basic_{target_function}" class="tab-btn active-tab">Básico</button>
            <button onclick="showTab_{target_function}('trig')" id="tab_trig_{target_function}" class="tab-btn">Trigonométricas</button>
            <button onclick="showTab_{target_function}('adv')" id="tab_adv_{target_function}" class="tab-btn">Avanzadas</button>
        </div>
        
        <!-- Panel Básico -->
        <div id="panel_basic_{target_function}" class="function-panel" style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 5px;">
            <button onclick="addToExpression_{target_function}('1')" class="math-btn">1</button>
            <button onclick="addToExpression_{target_function}('2')" class="math-btn">2</button>
            <button onclick="addToExpression_{target_function}('3')" class="math-btn">3</button>
            <button onclick="addToExpression_{target_function}('+')" class="math-btn">+</button>
            <button onclick="addToExpression_{target_function}('-')" class="math-btn">−</button>
            <button onclick="addToExpression_{target_function}('*')" class="math-btn">×</button>
            <button onclick="addToExpression_{target_function}('/')" class="math-btn">÷</button>
            <button onclick="clearExpression_{target_function}()" class="math-btn clear-btn">C</button>
            
            <button onclick="addToExpression_{target_function}('4')" class="math-btn">4</button>
            <button onclick="addToExpression_{target_function}('5')" class="math-btn">5</button>
            <button onclick="addToExpression_{target_function}('6')" class="math-btn">6</button>
            <button onclick="addToExpression_{target_function}('(')" class="math-btn">(</button>
            <button onclick="addToExpression_{target_function}(')')" class="math-btn">)</button>
            <button onclick="addToExpression_{target_function}('**')" class="math-btn">x^n</button>
            <button onclick="addToExpression_{target_function}('sqrt(')" class="math-btn">√</button>
            <button onclick="deleteLastChar_{target_function}()" class="math-btn delete-btn">⌫</button>
            
            <button onclick="addToExpression_{target_function}('7')" class="math-btn">7</button>
            <button onclick="addToExpression_{target_function}('8')" class="math-btn">8</button>
            <button onclick="addToExpression_{target_function}('9')" class="math-btn">9</button>
            <button onclick="addToExpression_{target_function}('exp(')" class="math-btn">e^x</button>
            <button onclick="addToExpression_{target_function}('log(')" class="math-btn">ln</button>
            <button onclick="addToExpression_{target_function}('log10(')" class="math-btn">log</button>
            <button onclick="addToExpression_{target_function}('abs(')" class="math-btn">|x|</button>
            <button onclick="copyExpression_{target_function}()" class="math-btn copy-btn">📋</button>
            
            <button onclick="addToExpression_{target_function}('0')" class="math-btn">0</button>
            <button onclick="addToExpression_{target_function}('.')" class="math-btn">.</button>
            <button onclick="addToExpression_{target_function}('x')" class="math-btn var-btn">x</button>
            <button onclick="addToExpression_{target_function}('pi')" class="math-btn">π</button>
            <button onclick="addToExpression_{target_function}('e')" class="math-btn">e</button>
            <button onclick="addToExpression_{target_function}('log2(')" class="math-btn">log₂</button>
            <button onclick="addToExpression_{target_function}('pow(')" class="math-btn">pow</button>
            <button onclick="addToExpression_{target_function}(',')" class="math-btn">,</button>
        </div>
        
        <!-- Panel Trigonométricas -->
        <div id="panel_trig_{target_function}" class="function-panel" style="display: none; grid-template-columns: repeat(8, 1fr); gap: 5px;">
            <button onclick="addToExpression_{target_function}('sin(')" class="math-btn">sin</button>
            <button onclick="addToExpression_{target_function}('cos(')" class="math-btn">cos</button>
            <button onclick="addToExpression_{target_function}('tan(')" class="math-btn">tan</button>
            <button onclick="addToExpression_{target_function}('asin(')" class="math-btn">arcsin</button>
            <button onclick="addToExpression_{target_function}('acos(')" class="math-btn">arccos</button>
            <button onclick="addToExpression_{target_function}('atan(')" class="math-btn">arctan</button>
            <button onclick="addToExpression_{target_function}('(')" class="math-btn">(</button>
            <button onclick="addToExpression_{target_function}(')')" class="math-btn">)</button>
            
            <button onclick="addToExpression_{target_function}('sinh(')" class="math-btn">sinh</button>
            <button onclick="addToExpression_{target_function}('cosh(')" class="math-btn">cosh</button>
            <button onclick="addToExpression_{target_function}('tanh(')" class="math-btn">tanh</button>
            <button onclick="addToExpression_{target_function}('asinh(')" class="math-btn">arcsinh</button>
            <button onclick="addToExpression_{target_function}('acosh(')" class="math-btn">arccosh</button>
            <button onclick="addToExpression_{target_function}('atanh(')" class="math-btn">arctanh</button>
            <button onclick="addToExpression_{target_function}('pi')" class="math-btn">π</button>
            <button onclick="addToExpression_{target_function}('x')" class="math-btn var-btn">x</button>
            
            <button onclick="addToExpression_{target_function}('*')" class="math-btn">×</button>
            <button onclick="addToExpression_{target_function}('+')" class="math-btn">+</button>
            <button onclick="addToExpression_{target_function}('-')" class="math-btn">−</button>
            <button onclick="addToExpression_{target_function}('**')" class="math-btn">^</button>
            <button onclick="addToExpression_{target_function}('sqrt(')" class="math-btn">√</button>
            <button onclick="addToExpression_{target_function}('abs(')" class="math-btn">|x|</button>
            <button onclick="deleteLastChar_{target_function}()" class="math-btn delete-btn">⌫</button>
            <button onclick="clearExpression_{target_function}()" class="math-btn clear-btn">C</button>
        </div>
        
        <!-- Panel Avanzadas -->
        <div id="panel_adv_{target_function}" class="function-panel" style="display: none; grid-template-columns: repeat(8, 1fr); gap: 5px;">
            <button onclick="addToExpression_{target_function}('gamma(')" class="math-btn">Γ</button>
            <button onclick="addToExpression_{target_function}('factorial(')" class="math-btn">n!</button>
            <button onclick="addToExpression_{target_function}('ceil(')" class="math-btn">⌈x⌉</button>
            <button onclick="addToExpression_{target_function}('floor(')" class="math-btn">⌊x⌋</button>
            <button onclick="addToExpression_{target_function}('fmod(')" class="math-btn">mod</button>
            <button onclick="addToExpression_{target_function}('degrees(')" class="math-btn">deg</button>
            <button onclick="addToExpression_{target_function}('radians(')" class="math-btn">rad</button>
            <button onclick="addToExpression_{target_function}('(')" class="math-btn">(</button>
            
            <button onclick="addToExpression_{target_function}('hypot(')" class="math-btn">hypot</button>
            <button onclick="addToExpression_{target_function}('atan2(')" class="math-btn">atan2</button>
            <button onclick="addToExpression_{target_function}('pow(')" class="math-btn">pow</button>
            <button onclick="addToExpression_{target_function}('gcd(')" class="math-btn">gcd</button>
            <button onclick="addToExpression_{target_function}('lcm(')" class="math-btn">lcm</button>
            <button onclick="addToExpression_{target_function}('copysign(')" class="math-btn">±</button>
            <button onclick="addToExpression_{target_function}('fabs(')" class="math-btn">|x|</button>
            <button onclick="addToExpression_{target_function}(')')" class="math-btn">)</button>
            
            <button onclick="addToExpression_{target_function}(',')" class="math-btn">,</button>
            <button onclick="addToExpression_{target_function}('x')" class="math-btn var-btn">x</button>
            <button onclick="addToExpression_{target_function}('pi')" class="math-btn">π</button>
            <button onclick="addToExpression_{target_function}('e')" class="math-btn">e</button>
            <button onclick="addToExpression_{target_function}('tau')" class="math-btn">τ</button>
            <button onclick="addToExpression_{target_function}('inf')" class="math-btn">∞</button>
            <button onclick="deleteLastChar_{target_function}()" class="math-btn delete-btn">⌫</button>
            <button onclick="clearExpression_{target_function}()" class="math-btn clear-btn">C</button>
        </div>
        
        <!-- Botón para enviar a Streamlit -->
        <div style="margin-top: 15px; text-align: center;">
            <button onclick="sendToStreamlit_{target_function}()" class="math-btn" style="background: #007bff; color: white; width: 100%;">
                ➤ Aplicar a {target_function}
            </button>
        </div>
        
        <style>
            .math-btn {{
                background: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                cursor: pointer;
                font-size: 12px;
                font-weight: bold;
                transition: all 0.2s;
                min-height: 35px;
            }}
            .math-btn:hover {{
                background: #e9ecef;
                border-color: #adb5bd;
                transform: translateY(-1px);
            }}
            .clear-btn {{ background: #dc3545 !important; color: white !important; }}
            .delete-btn {{ background: #ffc107 !important; }}
            .var-btn {{ background: #28a745 !important; color: white !important; }}
            .copy-btn {{ background: #17a2b8 !important; color: white !important; }}
            .tab-btn {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 8px 16px;
                cursor: pointer;
                border-radius: 4px 4px 0 0;
                margin-right: 5px;
                font-size: 14px;
            }}
            .active-tab {{
                background: #007bff !important;
                color: white !important;
            }}
            .function-panel {{
                min-height: 200px;
                padding: 10px 0;
            }}
        </style>
        
        <script>
            // Variable global para esta instancia del teclado
            if (typeof window.expressions === 'undefined') {{
                window.expressions = {{}};
            }}
            
            // Inicializar la expresión para este target
            if (!window.expressions['{target_function}']) {{
                window.expressions['{target_function}'] = '';
            }}
            
            // Mostrar/ocultar tabs
            function showTab_{target_function}(tabName) {{
                // Ocultar todos los paneles
                const panels = ['basic', 'trig', 'adv'];
                panels.forEach(panel => {{
                    const panelElement = document.getElementById('panel_' + panel + '_{target_function}');
                    const tabElement = document.getElementById('tab_' + panel + '_{target_function}');
                    if (panelElement) panelElement.style.display = 'none';
                    if (tabElement) tabElement.classList.remove('active-tab');
                }});
                
                // Mostrar panel seleccionado
                const selectedPanel = document.getElementById('panel_' + tabName + '_{target_function}');
                const selectedTab = document.getElementById('tab_' + tabName + '_{target_function}');
                if (selectedPanel) selectedPanel.style.display = 'grid';
                if (selectedTab) selectedTab.classList.add('active-tab');
            }}
            
            // Funciones específicas para este target
            function addToExpression_{target_function}(value) {{
                window.expressions['{target_function}'] += value;
                updateDisplay_{target_function}();
            }}
            
            function deleteLastChar_{target_function}() {{
                window.expressions['{target_function}'] = window.expressions['{target_function}'].slice(0, -1);
                updateDisplay_{target_function}();
            }}
            
            function clearExpression_{target_function}() {{
                window.expressions['{target_function}'] = '';
                updateDisplay_{target_function}();
            }}
            
            function updateDisplay_{target_function}() {{
                const displayElement = document.getElementById('display_{target_function}');
                if (displayElement) {{
                    displayElement.textContent = window.expressions['{target_function}'] || '';
                }}
            }}
            
            function copyExpression_{target_function}() {{
                const expr = window.expressions['{target_function}'];
                if (navigator.clipboard && expr) {{
                    navigator.clipboard.writeText(expr).then(() => {{
                        alert('Expresión copiada al portapapeles: ' + expr);
                    }}).catch(() => {{
                        // Fallback para navegadores que no soportan clipboard API
                        const textArea = document.createElement('textarea');
                        textArea.value = expr;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                        alert('Expresión copiada: ' + expr);
                    }});
                }} else if (expr) {{
                    alert('Expresión actual: ' + expr);
                }}
            }}
            
            // Función para enviar la expresión a Streamlit
            function sendToStreamlit_{target_function}() {{
                const expr = window.expressions['{target_function}'] || '';
                
                // Crear un elemento de entrada oculto para enviar datos a Streamlit
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'keyboard_input';
                input.value = JSON.stringify({{
                    target: '{target_function}',
                    expression: expr
                }});
                
                // Enviar el formulario
                const form = document.createElement('form');
                form.method = 'post';
                form.action = '';
                form.appendChild(input);
                document.body.appendChild(form);
                form.submit();
            }}
            
            // Función para establecer valor inicial
            function setInitialValue_{target_function}(value) {{
                window.expressions['{target_function}'] = value || '';
                updateDisplay_{target_function}();
            }}
            
            // Auto-inicializar display y mostrar tab básico
            setTimeout(() => {{
                updateDisplay_{target_function}();
                showTab_{target_function}('basic');
            }}, 100);
            
            // Event listener para teclas del teclado físico
            document.addEventListener('keydown', function(event) {{
                // Solo actuar si el teclado está visible y enfocado
                const keyboardElement = document.getElementById('{keyboard_id}');
                if (!keyboardElement) return;
                
                // Mapear teclas físicas a funciones del teclado
                const keyMap = {{
                    '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
                    '6': '6', '7': '7', '8': '8', '9': '9', '0': '0',
                    '+': '+', '-': '-', '*': '*', '/': '/',
                    '(': '(', ')': ')', '.': '.',
                    'x': 'x', 'X': 'x',
                    'Backspace': 'delete',
                    'Delete': 'clear',
                    'Escape': 'clear'
                }};
                
                if (keyMap[event.key]) {{
                    event.preventDefault();
                    if (event.key === 'Backspace') {{
                        deleteLastChar_{target_function}();
                    }} else if (event.key === 'Delete' || event.key === 'Escape') {{
                        clearExpression_{target_function}();
                    }} else {{
                        addToExpression_{target_function}(keyMap[event.key]);
                    }}
                }}
            }});
        </script>
    </div>
    """

def export_results_to_json(results: List[MethodResult]) -> str:
    """Exporta resultados a JSON."""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'results': [result.to_dict() for result in results]
    }
    return json.dumps(export_data, indent=2)

# Configuración de página
st.set_page_config(
    page_title="Métodos Numéricos con Teclado Matemático",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyectar MathJax
components.html(MATHJAX_CONFIG, height=0)

# CSS personalizado
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.success-metric {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
}

.warning-metric {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
}

.error-metric {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🔬 Métodos Numéricos con Teclado Matemático")
st.markdown("*Análisis completo con entrada intuitiva y visualización matemática*")

# Inicializar session state
if 'fx_expression' not in st.session_state:
    st.session_state.fx_expression = "x**2 - 2"
if 'dfx_expression' not in st.session_state:
    st.session_state.dfx_expression = "2*x"
if 'gx_expression' not in st.session_state:
    st.session_state.gx_expression = "(x + 2/x)/2"

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Presets de funciones
    function_presets = {
        "Personalizada": {"fx": "", "dfx": "", "gx": ""},
        "√2 (x² - 2)": {"fx": "x**2 - 2", "dfx": "2*x", "gx": "(x + 2/x)/2"},
        "e^x = 2x": {"fx": "exp(x) - 2*x", "dfx": "exp(x) - 2", "gx": "exp(x)/2"},
        "sin(x) = cos(x)": {"fx": "sin(x) - cos(x)", "dfx": "cos(x) + sin(x)", "gx": "asin(cos(x))"},
        "x³ - x - 1": {"fx": "x**3 - x - 1", "dfx": "3*x**2 - 1", "gx": "(x + 1)**(1/3)"},
        "Arco seno": {"fx": "asin(x) - pi/6", "dfx": "1/sqrt(1-x**2)", "gx": "sin(pi/6)"},
        "Polinomio 4°": {"fx": "x**4 - 2*x**2 - 5", "dfx": "4*x**3 - 4*x", "gx": "(2*x**2 + 5)**(1/4)"}
    }
    
    selected_preset = st.selectbox("Presets de funciones:", list(function_presets.keys()))
    
    if selected_preset != "Personalizada":
        preset = function_presets[selected_preset]
        st.session_state.fx_expression = preset["fx"]
        st.session_state.dfx_expression = preset["dfx"]
        st.session_state.gx_expression = preset["gx"]
    
    # Editor de funciones
    st.subheader("📝 Editor de Funciones")
    
    # f(x)
    st.markdown("**Función f(x):**")
    fx_new = st.text_input("f(x) =", value=st.session_state.fx_expression, key="fx_input")
    if fx_new != st.session_state.fx_expression:
        st.session_state.fx_expression = fx_new
    
    # Validación f(x)
    if st.session_state.fx_expression:
        is_valid, message, _ = FunctionValidator.validate_expression(st.session_state.fx_expression)
        if is_valid:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
    
    # f'(x)
    st.markdown("**Derivada f'(x) (opcional):**")
    dfx_new = st.text_input("f'(x) =", value=st.session_state.dfx_expression, key="dfx_input")
    if dfx_new != st.session_state.dfx_expression:
        st.session_state.dfx_expression = dfx_new
    
    # Validación f'(x)
    if st.session_state.dfx_expression:
        is_valid, message, _ = FunctionValidator.validate_expression(st.session_state.dfx_expression)
        if is_valid:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
    
    # g(x)
    st.markdown("**Función g(x) para punto fijo:**")
    gx_new = st.text_input("g(x) =", value=st.session_state.gx_expression, key="gx_input")
    if gx_new != st.session_state.gx_expression:
        st.session_state.gx_expression = gx_new
    
    # Validación g(x)
    if st.session_state.gx_expression:
        is_valid, message, _ = FunctionValidator.validate_expression(st.session_state.gx_expression)
        if is_valid:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
    
    # Selección de métodos
    st.subheader("🎯 Métodos a Ejecutar")
    methods_to_run = st.multiselect(
        "Selecciona métodos:",
        ["Newton-Raphson", "Secante", "Bisección", "Punto Fijo", "Aitken"],
        default=["Newton-Raphson", "Secante", "Bisección"]
    )
    
    # Parámetros numéricos
    st.subheader("🔧 Parámetros")
    
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀:", value=1.5, format="%.6f")
        x1 = st.number_input("x₁ (Secante):", value=2.0, format="%.6f")
    with col2:
        a = st.number_input("a (Bisección):", value=0.0, format="%.6f")
        b = st.number_input("b (Bisección):", value=2.0, format="%.6f")
    
    tol = st.number_input("Tolerancia:", value=1e-8, format="%.2e")
    max_iter = st.number_input("Max iteraciones:", value=50, min_value=1, max_value=200)

# Área principal
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("🔢 Teclado Matemático")
    
    # Selector de función a editar
    current_editing = st.selectbox(
        "Función a editar:",
        ["f(x) - Función principal", "f'(x) - Derivada", "g(x) - Punto fijo"]
    )
    
    # Mostrar teclado
    if "f(x)" in current_editing:
        target = "fx"
    elif "f'(x)" in current_editing:
        target = "dfx"
    else:
        target = "gx"
    
    keyboard_html = create_math_keyboard(target)
    components.html(keyboard_html, height=400)
    
    # Botones de acción
    st.subheader("⚡ Acciones Rápidas")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Probar f(x)", use_container_width=True):
            if st.session_state.fx_expression:
                try:
                    _, _, f_func = FunctionValidator.validate_expression(st.session_state.fx_expression)
                    if f_func:
                        result = f_func(1.0)
                        st.success(f"f(1.0) = {result:.6f}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("📋 Limpiar Todo", use_container_width=True):
            st.session_state.fx_expression = ""
            st.session_state.dfx_expression = ""
            st.session_state.gx_expression = ""
            st.rerun()
    
    # Sección de derivadas
    st.subheader("🔄 Calculadora de Derivadas")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("∂ Calcular f'(x)", use_container_width=True):
            if st.session_state.fx_expression:
                try:
                    derivative = SymbolicDerivative.calculate_derivative(st.session_state.fx_expression)
                    st.session_state.dfx_expression = derivative
                    st.success(f"✅ Derivada calculada: {derivative}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error calculando derivada: {str(e)}")
    
    with col2:
        if st.button("🔄 Auto g(x)", use_container_width=True):
            if st.session_state.fx_expression:
                expr = st.session_state.fx_expression
                try:
                    if "x**2" in expr and "- 2" in expr:
                        st.session_state.gx_expression = "(x + 2/x)/2"
                    elif "exp(x)" in expr and "2*x" in expr:
                        st.session_state.gx_expression = "exp(x)/2"
                    elif "sin(x)" in expr and "cos(x)" in expr:
                        st.session_state.gx_expression = "x + sin(x) - cos(x)"
                    else:
                        st.session_state.gx_expression = f"x - ({expr})/10"
                    
                    st.success("✅ Función g(x) generada automáticamente")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ No se pudo generar g(x) automáticamente")
    
    # Mostrar información de la función seleccionada
    if "f(x)" in current_editing and st.session_state.fx_expression:
        st.info(f"📝 Editando: f(x) = {st.session_state.fx_expression}")
    elif "f'(x)" in current_editing and st.session_state.dfx_expression:
        st.info(f"📝 Editando: f'(x) = {st.session_state.dfx_expression}")
    elif "g(x)" in current_editing and st.session_state.gx_expression:
        st.info(f"📝 Editando: g(x) = {st.session_state.gx_expression}")

with col_right:
    st.header("📊 Vista Matemática")
    
    # Renderizar expresiones matemáticas
    if st.session_state.fx_expression:
        st.subheader("Función Principal")
        MathRenderer.render_expression(st.session_state.fx_expression, "f(x)")
    
    if st.session_state.dfx_expression:
        st.subheader("Derivada")
        MathRenderer.render_expression(st.session_state.dfx_expression, "f'(x)")
    
    if st.session_state.gx_expression:
        st.subheader("Función de Punto Fijo")
        MathRenderer.render_expression(st.session_state.gx_expression, "g(x)")
    
    # Mostrar fórmulas de métodos
    if methods_to_run:
        st.subheader("Fórmulas de Métodos")
        for method in methods_to_run:
            MathRenderer.render_method_formula(method)

# Botón principal
st.header("🚀 Ejecutar Análisis")

if st.button("🔬 Ejecutar Análisis Completo", type="primary", use_container_width=True):
    
    fx_expr = st.session_state.fx_expression
    dfx_expr = st.session_state.dfx_expression
    gx_expr = st.session_state.gx_expression
    
    if not methods_to_run:
        st.error("❌ Selecciona al menos un método para ejecutar")
        st.stop()
    
    if not fx_expr:
        st.error("❌ Debes proporcionar la función f(x)")
        st.stop()
    
    # Validar funciones
    is_valid_f, msg_f, f_func = FunctionValidator.validate_expression(fx_expr)
    if not is_valid_f:
        st.error(f"❌ Error en f(x): {msg_f}")
        st.stop()
    
    df_func = None
    if dfx_expr:
        is_valid_df, msg_df, df_func = FunctionValidator.validate_expression(dfx_expr)
        if not is_valid_df:
            st.warning(f"⚠️ Error en f'(x): {msg_df}. Se usará derivada numérica.")
    
    g_func = None
    if gx_expr and ("Punto Fijo" in methods_to_run or "Aitken" in methods_to_run):
        is_valid_g, msg_g, g_func = FunctionValidator.validate_expression(gx_expr)
        if not is_valid_g:
            st.error(f"❌ Error en g(x): {msg_g}")
            st.stop()
    
    # Ejecutar métodos
    results = []
    
    with st.spinner("🔄 Ejecutando métodos numéricos..."):
        progress_bar = st.progress(0)
        total_methods = len(methods_to_run)
        
        if "Newton-Raphson" in methods_to_run:
            result = NumericalMethods.newton_raphson_adaptive(f_func, x0, df_func, tol, max_iter)
            results.append(result)
            progress_bar.progress(len(results) / total_methods)
        
        if "Secante" in methods_to_run:
            result = NumericalMethods.secant_method(f_func, x0, x1, tol, max_iter)
            results.append(result)
            progress_bar.progress(len(results) / total_methods)
        
        if "Bisección" in methods_to_run:
            result = NumericalMethods.bisection_method(f_func, a, b, tol, max_iter)
            results.append(result)
            progress_bar.progress(len(results) / total_methods)
        
        if "Punto Fijo" in methods_to_run:
            if g_func:
                result = NumericalMethods.punto_fijo(g_func, x0, tol, max_iter)
                results.append(result)
                progress_bar.progress(len(results) / total_methods)
            else:
                st.error("❌ Se necesita función g(x) para Punto Fijo")
        
        if "Aitken" in methods_to_run:
            if g_func:
                result = NumericalMethods.punto_fijo_aitken(g_func, x0, tol, max_iter)
                results.append(result)
                progress_bar.progress(len(results) / total_methods)
            else:
                st.error("❌ Se necesita función g(x) para Aitken")
    
    st.success("✅ Análisis completado")
    
    # Mostrar función analizada
    st.header("🎯 Función Analizada")
    col1, col2 = st.columns(2)
    
    with col1:
        MathRenderer.render_expression(fx_expr, "f(x)")
    
    with col2:
        if dfx_expr:
            MathRenderer.render_expression(dfx_expr, "f'(x)")
        else:
            st.info("Se usó derivada numérica automática")
        
        if gx_expr:
            MathRenderer.render_expression(gx_expr, "g(x)")

    # Resultados principales
    st.header("📊 Resultados del Análisis")
    
    # Métricas principales
    successful_results = [r for r in results if r.root is not None]
    
    if successful_results:
        col1, col2, col3, col4 = st.columns(4)
        
        fastest = min(successful_results, key=lambda x: x.execution_time)
        most_accurate = min(successful_results, key=lambda x: x.final_error)
        least_iterations = min(successful_results, key=lambda x: x.iterations)
        
        with col1:
            st.metric("🏃 Más Rápido", fastest.name, f"{fastest.execution_time:.6f}s")
        
        with col2:
            st.metric("🎯 Más Preciso", most_accurate.name, f"{most_accurate.final_error:.2e}")
        
        with col3:
            st.metric("⚡ Menos Iteraciones", least_iterations.name, f"{least_iterations.iterations}")
        
        with col4:
            success_rate = len(successful_results) / len(results) * 100
            st.metric("✅ Tasa de Éxito", f"{success_rate:.0f}%", f"{len(successful_results)}/{len(results)}")
    
    # Tabla de comparación
    st.subheader("📋 Tabla Comparativa")
    comparison_data = []
    
    for result in results:
        comparison_data.append({
            'Método': result.name,
            'Raíz': f"{result.root:.10f}" if result.root is not None else "No convergió",
            'Estado': result.status,
            'Iteraciones': result.iterations,
            'Tiempo (s)': f"{result.execution_time:.6f}",
            'Error Final': f"{result.final_error:.2e}" if result.final_error != float('inf') else "∞",
            'Convergencia': result.convergence_rate or "N/A"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Visualizaciones
    if successful_results:
        # Gráfico de convergencia
        st.subheader("📈 Análisis de Convergencia")
        conv_fig = AdvancedVisualizations.create_interactive_convergence_plot(successful_results)
        st.plotly_chart(conv_fig, use_container_width=True)
        
        # Gráfico de función
        st.subheader("🎯 Función y Raíces")
        func_fig = AdvancedVisualizations.create_function_plot_with_roots(f_func, successful_results)
        st.plotly_chart(func_fig, use_container_width=True)
        
        # Gráfico de rendimiento
        st.subheader("⚡ Análisis de Rendimiento")
        perf_fig = AdvancedVisualizations.create_performance_comparison(successful_results)
        st.plotly_chart(perf_fig, use_container_width=True)
    
    # Detalles por método
    st.header("📋 Detalles por Método")
    
    for result in results:
        with st.expander(f"{result.name} - {result.status}"):
            # Mostrar fórmula del método
            MathRenderer.render_method_formula(result.name)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if result.history:
                    # Crear DataFrame según el método
                    if result.name == "Newton-Raphson":
                        df_detail = pd.DataFrame(result.history, 
                                               columns=['n', 'x_n', 'f(x_n)', "f'(x_n)", 'Error Abs', 'Error Rel'])
                    elif result.name == "Secante":
                        df_detail = pd.DataFrame(result.history,
                                               columns=['n', 'x_0', 'x_1', 'f(x_0)', 'f(x_1)', 'Error Abs', 'Error Rel'])
                    elif result.name == "Bisección":
                        df_detail = pd.DataFrame(result.history,
                                               columns=['n', 'a', 'b', 'c', 'f(a)', 'f(b)', 'f(c)', 'Error Abs', 'Error Rel'])
                    elif result.name in ["Punto Fijo", "Aitken"]:
                        df_detail = pd.DataFrame(result.history,
                                               columns=['n', 'x_n', 'x_{n+1}', 'Error Abs', 'Error Rel'])
                    
                    # Formatear números
                    for col in df_detail.columns:
                        if col != 'n':
                            df_detail[col] = df_detail[col].apply(
                                lambda x: f"{x:.6g}" if isinstance(x, (int, float)) else x
                            )
                    
                    st.dataframe(df_detail, use_container_width=True)
                else:
                    st.warning("No hay datos de iteraciones disponibles")
            
            with col2:
                # Métricas individuales
                if result.root is not None:
                    st.markdown('<div class="metric-container success-metric">', unsafe_allow_html=True)
                    st.metric("Raíz Encontrada", f"{result.root:.8f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-container error-metric">', unsafe_allow_html=True)
                    st.metric("Estado", "No convergió")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.metric("Tiempo", f"{result.execution_time:.6f}s")
                st.metric("Iteraciones", result.iterations)
                
                if result.final_error != float('inf'):
                    st.metric("Error Final", f"{result.final_error:.2e}")
                
                if result.convergence_rate:
                    st.metric("Convergencia", result.convergence_rate)
    
    # Exportación
    st.header("💾 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        json_data = export_results_to_json(results)
        st.download_button(
            label="📥 Descargar JSON",
            data=json_data,
            file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        csv_data = df_comparison.to_csv(index=False)
        st.download_button(
            label="📊 Descargar CSV",
            data=csv_data,
            file_name=f"comparacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Página de inicio
    if not st.session_state.fx_expression:
        st.info("👆 Usa el **teclado matemático** para crear funciones o selecciona un **preset** en el panel lateral")
    
    # Ejemplos interactivos
    st.header("📚 Ejemplos Interactivos")
    
    example_tabs = st.tabs(["√2", "Exponencial", "Trigonométrica", "Trigonométrica Inversa", "Punto Fijo"])
    
    with example_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Encontrar √2")
            st.markdown("**Problema**: Resolver $x^2 - 2 = 0$")
            st.code("f(x) = x**2 - 2")
            st.code("f'(x) = 2*x")
            st.code("g(x) = (x + 2/x)/2")
            if st.button("Cargar ejemplo √2"):
                st.session_state.fx_expression = "x**2 - 2"
                st.session_state.dfx_expression = "2*x"
                st.session_state.gx_expression = "(x + 2/x)/2"
                st.rerun()
        with col2:
            MathRenderer.render_expression("x**2 - 2", "f(x)")
    
    with example_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Ecuación Exponencial")
            st.markdown("**Problema**: Resolver $e^x = 2x$")
            st.code("f(x) = exp(x) - 2*x")
            st.code("f'(x) = exp(x) - 2")
            st.code("g(x) = exp(x)/2")
            if st.button("Cargar ejemplo exponencial"):
                st.session_state.fx_expression = "exp(x) - 2*x"
                st.session_state.dfx_expression = "exp(x) - 2"
                st.session_state.gx_expression = "exp(x)/2"
                st.rerun()
        with col2:
            MathRenderer.render_expression("exp(x) - 2*x", "f(x)")
    
    with example_tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Función Trigonométrica")
            st.markdown("**Problema**: Resolver $\\sin(x) = \\cos(x)$")
            st.code("f(x) = sin(x) - cos(x)")
            st.code("f'(x) = cos(x) + sin(x)")
            st.code("g(x) = x + sin(x) - cos(x)")
            if st.button("Cargar ejemplo trigonométrico"):
                st.session_state.fx_expression = "sin(x) - cos(x)"
                st.session_state.dfx_expression = "cos(x) + sin(x)"
                st.session_state.gx_expression = "x + sin(x) - cos(x)"
                st.rerun()
        with col2:
            MathRenderer.render_expression("sin(x) - cos(x)", "f(x)")
    
    with example_tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Función Trigonométrica Inversa")
            st.markdown("**Problema**: Resolver $\\arcsin(x) = \\frac{\\pi}{6}$")
            st.code("f(x) = asin(x) - pi/6")
            st.code("f'(x) = 1/sqrt(1-x**2)")
            st.code("g(x) = sin(pi/6)")
            if st.button("Cargar ejemplo arcseno"):
                st.session_state.fx_expression = "asin(x) - pi/6"
                st.session_state.dfx_expression = "1/sqrt(1-x**2)"
                st.session_state.gx_expression = "sin(pi/6)"
                st.rerun()
        with col2:
            MathRenderer.render_expression("asin(x) - pi/6", "f(x)")
    
    with example_tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Ejemplo de Punto Fijo")
            st.markdown("**Problema**: Resolver $x = \\cos(x)$")
            st.code("f(x) = x - cos(x)")
            st.code("f'(x) = 1 + sin(x)")
            st.code("g(x) = cos(x)")
            if st.button("Cargar ejemplo punto fijo"):
                st.session_state.fx_expression = "x - cos(x)"
                st.session_state.dfx_expression = "1 + sin(x)"
                st.session_state.gx_expression = "cos(x)"
                st.rerun()
        with col2:
            MathRenderer.render_expression("cos(x)", "g(x)")
    
    # Tutorial del teclado
    st.header("🎓 Guía del Teclado Matemático Avanzado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔢 Funciones del Teclado
        
        **Panel Básico:**
        - `1-9, 0, .` - Números y decimal
        - `+, -, ×, ÷, ^` - Operaciones básicas
        - `√, e^x, ln, log, log₂` - Funciones elementales
        - `π, e, τ, ∞` - Constantes matemáticas
        
        **Panel Trigonométrico:**
        - `sin, cos, tan` - Funciones trigonométricas
        - `arcsin, arccos, arctan` - **Funciones inversas**
        - `sinh, cosh, tanh` - Funciones hiperbólicas
        - `arcsinh, arccosh, arctanh` - **Inversas hiperbólicas**
        
        **Panel Avanzado:**
        - `Γ, n!, ⌈x⌉, ⌊x⌋` - Funciones especiales
        - `mod, gcd, lcm` - Aritmética modular
        - `hypot, atan2` - Funciones geométricas
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Nuevas Características
        
        **Múltiples Paneles:**
        - **Básico**: Operaciones fundamentales
        - **Trigonométrico**: Funciones trig e inversas
        - **Avanzado**: Funciones matemáticas especiales
        
        **Derivadas Automáticas:**
        - Usa **SymPy** para cálculo simbólico
        - Calcula derivadas de cualquier función
        - Soporte para funciones trigonométricas inversas
        
        **Soporte para 3 Funciones:**
        - `f(x)` - Función principal
        - `f'(x)` - Derivada (calculable automáticamente)
        - `g(x)` - Función para punto fijo
        
        **5 Métodos Numéricos:**
        - Newton-Raphson, Secante, Bisección
        - **Punto Fijo** y **Aitken** agregados
        """)
    
    # Información adicional
    with st.expander("📖 Documentación de Nuevos Métodos"):
        st.markdown("""
        ## 🎯 Nuevos Métodos Implementados
        
        ### Punto Fijo
        - **Fórmula**: $x_{n+1} = g(x_n)$
        - **Condición**: $|g'(x)| < 1$ cerca de la raíz para convergencia
        - **Uso**: Reformular $f(x) = 0$ como $x = g(x)$
        - **Ejemplo**: Para $x^2 - 2 = 0$, usar $g(x) = 2/x$ o $g(x) = (x + 2/x)/2$
        
        ### Aitken (Aceleración)
        - **Fórmula**: $x_{acc} = x_2 - \\frac{(x_2 - x_1)^2}{x_2 - 2x_1 + x_0}$
        - **Propósito**: Acelerar la convergencia del método de punto fijo
        - **Ventaja**: Puede convertir convergencia lineal en super-lineal
        - **Uso**: Automático cuando seleccionas "Aitken"
        
        ## 🧮 Funciones Trigonométricas Inversas
        
        **Disponibles en el teclado:**
        - `asin(x)` → $\\arcsin(x)$ (arcseno)
        - `acos(x)` → $\\arccos(x)$ (arccoseno)
        - `atan(x)` → $\\arctan(x)$ (arcotangente)
        - `asinh(x)` → $\\text{arcsinh}(x)$ (arcseno hiperbólico)
        - `acosh(x)` → $\\text{arccosh}(x)$ (arccoseno hiperbólico)
        - `atanh(x)` → $\\text{arctanh}(x)$ (arcotangente hiperbólica)
        
        **Dominios importantes:**
        - `asin(x)`: $x \\in [-1, 1]$
        - `acos(x)`: $x \\in [-1, 1]$
        - `atan(x)`: $x \\in \\mathbb{R}$
        - `acosh(x)`: $x \\geq 1$
        - `atanh(x)`: $x \\in (-1, 1)$
        
        ## 🔄 Calculadora de Derivadas Simbólicas
        
        **Powered by SymPy:**
        - Calcula derivadas exactas de cualquier función
        - Soporte completo para funciones trigonométricas inversas
        - Maneja funciones compuestas automáticamente
        - Simplifica expresiones resultantes
        
        **Ejemplos de uso:**
        - $\\frac{d}{dx}[\\arcsin(x)] = \\frac{1}{\\sqrt{1-x^2}}$
        - $\\frac{d}{dx}[\\arctan(x)] = \\frac{1}{1+x^2}$
        - $\\frac{d}{dx}[\\sinh^{-1}(x)] = \\frac{1}{\\sqrt{x^2+1}}$
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <h4>🔬 Métodos Numéricos Avanzados - Versión Completa</h4>
    <p>Herramienta educativa completa para análisis de métodos de búsqueda de raíces</p>
    <p><em>Desarrollado con Streamlit • MathJax • Plotly • SymPy • NumPy</em></p>
    <p><strong>Versión 4.0</strong> - 5 Métodos • Funciones Trigonométricas Inversas • Derivadas Simbólicas</p>
    <p>✨ <strong>Nuevas características:</strong> Punto Fijo • Aitken • Teclado de 3 paneles • Calculadora de derivadas</p>
</div>
""", unsafe_allow_html=True)