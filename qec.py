"""
FRAMEWORK AVANZADO DE CORRECCIÓN DE ERRORES CUÁNTICOS
Usando la Ecuación de Lindblad para Modelar Dinámicas de Error y Corrección

Este framework simula la evolución de un estado cuántico bajo la influencia
de un canal de error y aplica corrección periódica de errores.

Características principales:
- Códigos de corrección diversos (Shor, Bit-Flip).
- Procesos de detección y corrección dinámicos.
- Visualización de métricas de rendimiento cuantitativas como la fidelidad.
- Optimización de protocolos (frecuencia de corrección).

Author: [Tu nombre]
Date: [Fecha]
Version: 1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# CLASES BASE PARA EL FRAMEWORK
# ============================================================================

class ErrorType(Enum):
    """
    Enumeración de tipos de errores cuánticos soportados.
    
    Attributes:
        BIT_FLIP: Errores de flip de bit (X errors)
        PHASE_FLIP: Errores de flip de fase (Z errors)  
        DEPOLARIZING: Canal despolarizante
        AMPLITUDE_DAMPING: Amortiguamiento de amplitud
        PHASE_DAMPING: Amortiguamiento de fase
    """
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"  # Corregido: era "phase_dampin"


class CodeType(Enum):
    """
    Enumeración de tipos de códigos de corrección cuánticos.
    
    Attributes:
        REPETITION_3: Código de repetición de 3 qubits
        SHOR_9: Código de Shor de 9 qubits
        STEANE_7: Código de Steane de 7 qubits
        SURFACE: Códigos de superficie
        CUSTOM: Código personalizado
    """
    REPETITION_3 = "repetition_3"
    SHOR_9 = "shor_9"
    STEANE_7 = "steane_7"
    SURFACE = "surface"
    CUSTOM = "custom"


@dataclass
class CorrectionMetrics:
    """
    Métricas de rendimiento del código de corrección de errores.
    
    Attributes:
        logical_error_rate (float): Tasa de error lógico
        threshold_rate (float): Tasa umbral del código
        correction_efficiency (float): Eficiencia de corrección [0, 1]
        resource_overhead (int): Overhead de recursos físicos
        decoding_time (float): Tiempo de decodificación en segundos
    """
    logical_error_rate: float
    threshold_rate: float
    correction_efficiency: float
    resource_overhead: int
    decoding_time: float


@dataclass
class ErrorModel:
    """
    Modelo de error para el sistema cuántico.
    
    Attributes:
        error_type (ErrorType): Tipo de error cuántico
        error_rates (Dict[str, float]): Tasas de error por parámetro
        correlation_matrix (Optional[np.ndarray]): Matriz de correlaciones espaciales
        temporal_correlations (bool): Si hay correlaciones temporales
    """
    error_type: ErrorType
    error_rates: Dict[str, float]
    correlation_matrix: Optional[np.ndarray] = None
    temporal_correlations: bool = False


# ============================================================================
# GENERADORES DE CÓDIGOS DE CORRECCIÓN
# ============================================================================

class QuantumErrorCorrectionCode:
    """
    Clase base para códigos de corrección de errores cuánticos.
    
    Esta clase define la interfaz común para todos los códigos de corrección
    de errores cuánticos en el framework.
    
    Attributes:
        name (str): Nombre del código
        n_physical (int): Número de qubits físicos
        k_logical (int): Número de qubits lógicos
        distance (int): Distancia del código
        dimension (int): Dimensión del espacio de Hilbert
    """

    def __init__(self, name: str, n_physical: int, k_logical: int, distance: int):
        """
        Inicializa el código de corrección de errores.
        
        Args:
            name: Nombre identificativo del código
            n_physical: Número de qubits físicos requeridos
            k_logical: Número de qubits lógicos codificados
            distance: Distancia mínima del código
        """
        self.name = name
        self.n_physical = n_physical  # Qubits físicos
        self.k_logical = k_logical    # Qubits lógicos
        self.distance = distance      # Distancia del código
        self.dimension = 2**n_physical

    def get_stabilizers(self) -> List[np.ndarray]:
        """
        Retorna los operadores estabilizadores del código.
        
        Returns:
            List[np.ndarray]: Lista de matrices estabilizadoras
            
        Raises:
            NotImplementedError: Si no está implementado en la subclase
        """
        raise NotImplementedError("Debe implementarse en la subclase")

    def get_logical_operators(self) -> Dict[str, List[np.ndarray]]:
        """
        Retorna operadores lógicos X y Z del código.
        
        Returns:
            Dict[str, List[np.ndarray]]: Diccionario con operadores lógicos
                - 'X': Lista de operadores X lógicos
                - 'Z': Lista de operadores Z lógicos
                
        Raises:
            NotImplementedError: Si no está implementado en la subclase
        """
        raise NotImplementedError("Debe implementarse en la subclase")

    def encode_state(self, logical_state: np.ndarray) -> np.ndarray:
        """
        Codifica un estado lógico en el código.
        
        Transforma |ψ⟩ = α|0⟩ + β|1⟩ → α|0̄⟩ + β|1̄⟩
        donde |0̄⟩, |1̄⟩ son los estados de código.
        
        Args:
            logical_state: Estado lógico a codificar (vector de 2^k dimensiones)
            
        Returns:
            np.ndarray: Estado codificado (vector de 2^n dimensiones)
            
        Raises:
            NotImplementedError: Si no está implementado en la subclase
        """
        raise NotImplementedError("Debe implementarse en la subclase")

    def get_syndrome_operators(self) -> List[np.ndarray]:
        """
        Retorna operadores para medición de síndrome.
        
        Returns:
            List[np.ndarray]: Lista de operadores de síndrome
        """
        return self.get_stabilizers()


class ThreeQubitBitFlipCode(QuantumErrorCorrectionCode):
    """
    Código de corrección de bit-flip de 3 qubits.
    
    Este código protege contra errores de bit-flip (errores X) en un solo qubit
    usando redundancia. Codifica: |0⟩ → |000⟩, |1⟩ → |111⟩
    
    Parámetros del código: [3,1,3] - 3 qubits físicos, 1 lógico, distancia 3
    """

    def __init__(self):
        """Inicializa el código de 3 qubits para bit-flip."""
        super().__init__("3-Qubit Bit-Flip", 3, 1, 3)
        self._build_operators()

    def _build_operators(self) -> None:
        """
        Construye operadores de Pauli para 3 qubits.
        
        Crea las matrices X_i y Z_i para cada qubit i ∈ {0,1,2}
        usando productos tensoriales de matrices de Pauli.
        """
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])

        # Operadores individuales X_i para cada qubit
        self.X = [
            np.kron(X, np.kron(I, I)),  # X₀ ⊗ I₁ ⊗ I₂
            np.kron(I, np.kron(X, I)),  # I₀ ⊗ X₁ ⊗ I₂
            np.kron(I, np.kron(I, X))   # I₀ ⊗ I₁ ⊗ X₂
        ]

        # Operadores individuales Z_i para cada qubit
        self.Z = [
            np.kron(Z, np.kron(I, I)),  # Z₀ ⊗ I₁ ⊗ I₂
            np.kron(I, np.kron(Z, I)),  # I₀ ⊗ Z₁ ⊗ I₂
            np.kron(I, np.kron(I, Z))   # I₀ ⊗ I₁ ⊗ Z₂
        ]

    def get_stabilizers(self) -> List[np.ndarray]:
        """
        Retorna los estabilizadores del código.
        
        Para el código de 3 qubits bit-flip:
        - S₁ = X₀X₁ (detecta errores entre qubits 0 y 1)
        - S₂ = X₁X₂ (detecta errores entre qubits 1 y 2)
        
        Returns:
            List[np.ndarray]: Lista con los dos estabilizadores
        """
        return [
            self.X[0] @ self.X[1],  # X₀X₁
            self.X[1] @ self.X[2]   # X₁X₂
        ]

    def get_logical_operators(self) -> Dict[str, List[np.ndarray]]:
        """
        Retorna los operadores lógicos del código.
        
        Para el código de 3 qubits:
        - X_L = X₀X₁X₂ (flip lógico)
        - Z_L = Z₀ (fase lógica, cualquier Z_i funciona)
        
        Returns:
            Dict[str, List[np.ndarray]]: Operadores lógicos X y Z
        """
        X_L = self.X[0] @ self.X[1] @ self.X[2]
        Z_L = self.Z[0]

        return {
            'X': [X_L],
            'Z': [Z_L]
        }

    def encode_state(self, logical_state: np.ndarray) -> np.ndarray:
        """
        Codifica un estado lógico en el código de 3 qubits.
        
        Transformación: α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩
        
        Args:
            logical_state: Vector [α, β] del estado lógico
            
        Returns:
            np.ndarray: Estado codificado de 8 dimensiones
            
        Raises:
            ValueError: Si el estado lógico no tiene dimensión 2
        """
        if len(logical_state) != 2:
            raise ValueError("Estado lógico debe ser de 2 dimensiones")

        # Estados base del código
        psi_000 = np.zeros(8)
        psi_000[0] = 1.0  # |000⟩
        
        psi_111 = np.zeros(8)
        psi_111[7] = 1.0  # |111⟩

        # Combinación lineal
        encoded = logical_state[0] * psi_000 + logical_state[1] * psi_111
        return encoded / np.linalg.norm(encoded)


class ShorNineQubitCode(QuantumErrorCorrectionCode):
    """
    Código de Shor de 9 qubits para corrección completa de errores.
    
    Este código puede corregir cualquier error de un solo qubit (X, Y, Z).
    Es una concatenación del código de 3-qubits bit-flip con el código de
    3-qubits phase-flip.
    
    Parámetros del código: [9,1,3] - 9 qubits físicos, 1 lógico, distancia 3
    """

    def __init__(self):
        """Inicializa el código de Shor de 9 qubits."""
        super().__init__("Shor 9-Qubit", 9, 1, 3)
        self._build_operators()

    def _build_operators(self) -> None:
        """
        Construye operadores de Pauli para 9 qubits.
        
        Crea un diccionario con las matrices de Pauli básicas
        para usar en la construcción de operadores multi-qubit.
        """
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        Y = 1j * X @ Z

        self.pauli_ops = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    def _multi_qubit_pauli(self, pauli_string: str) -> np.ndarray:
        """
        Construye operador de Pauli multi-qubit desde string.
        
        Args:
            pauli_string: String de 9 caracteres con operadores Pauli
                         (ej: "ZZIIIIIII" para Z₀Z₁I₂...I₈)
            
        Returns:
            np.ndarray: Matriz del operador multi-qubit
            
        Raises:
            ValueError: Si el string no tiene 9 caracteres
        """
        if len(pauli_string) != 9:
            raise ValueError("String debe tener 9 caracteres")

        op = np.array([[1.0]])  # Inicializar como escalar
        for char in pauli_string:
            op = np.kron(op, self.pauli_ops[char])

        return op

    def get_stabilizers(self) -> List[np.ndarray]:
        """
        Retorna los 8 generadores estabilizadores del código de Shor.
        
        Incluye estabilizadores para detectar errores de fase (Z-type)
        dentro de cada bloque de 3 qubits, y errores de bit (X-type)
        entre bloques.
        
        Returns:
            List[np.ndarray]: Lista de 8 estabilizadores
        """
        # Z-type: detectan errores de fase dentro de cada bloque
        stabilizers = [
            self._multi_qubit_pauli("ZZIIIIIII"),  # Z₀Z₁
            self._multi_qubit_pauli("IZZIIIIII"),  # Z₁Z₂
            self._multi_qubit_pauli("IIIZZIIII"),  # Z₃Z₄
            self._multi_qubit_pauli("IIIIZZIII"),  # Z₄Z₅
            self._multi_qubit_pauli("IIIIIIZZI"),  # Z₆Z₇
            self._multi_qubit_pauli("IIIIIIIZZ"),  # Z₇Z₈
            
            # X-type: detectan errores de bit entre bloques
            self._multi_qubit_pauli("XXXXXXIII"),  # X₀X₁X₂X₃X₄X₅
            self._multi_qubit_pauli("IIIXXXXXX")   # X₃X₄X₅X₆X₇X₈
        ]

        return stabilizers

    def get_logical_operators(self) -> Dict[str, List[np.ndarray]]:
        """
        Retorna los operadores lógicos del código de Shor.
        
        Returns:
            Dict[str, List[np.ndarray]]: Operadores lógicos X y Z
        """
        X_L = self._multi_qubit_pauli("XXXXXXXXX")  # X en todos los qubits
        Z_L = self._multi_qubit_pauli("ZZZIIIZZI")  # Z en el primer qubit de cada bloque

        return {
            'X': [X_L],
            'Z': [Z_L]
        }

    def encode_state(self, logical_state: np.ndarray) -> np.ndarray:
        """
        Codifica estado lógico en el código de Shor.
        
        Esta es una representación simplificada del estado de Shor.
        La codificación real requiere un circuito cuántico específico.
        
        Args:
            logical_state: Vector [α, β] del estado lógico
            
        Returns:
            np.ndarray: Estado codificado de 512 dimensiones
            
        Raises:
            ValueError: Si el estado lógico no tiene dimensión 2
        """
        if len(logical_state) != 2:
            raise ValueError("Estado lógico debe ser de 2 dimensiones")

        # Estados de código base para cada bloque de 3 qubits
        # |±⟩ = (|000⟩ ± |111⟩)/√2
        ket0_block = np.zeros(8)
        ket0_block[0] = 1/np.sqrt(2)  # |000⟩
        ket0_block[7] = 1/np.sqrt(2)  # |111⟩
        
        ket1_block = np.zeros(8)
        ket1_block[0] = 1/np.sqrt(2)   # |000⟩
        ket1_block[7] = -1/np.sqrt(2)  # |111⟩

        # Estados lógicos de 9 qubits
        psi_0_L = np.kron(np.kron(ket0_block, ket0_block), ket0_block)
        psi_1_L = np.kron(np.kron(ket1_block, ket1_block), ket1_block)

        # Combinación lineal
        encoded = logical_state[0] * psi_0_L + logical_state[1] * psi_1_L
        return encoded / np.linalg.norm(encoded)


# ============================================================================
# SIMULADOR DE CORRECCIÓN DE ERRORES
# ============================================================================

class QuantumErrorCorrectionSimulator:
    """
    Simulador completo de corrección de errores cuánticos.
    
    Simula la evolución temporal de estados cuánticos bajo errores modelados
    por la ecuación de Lindblad, con corrección periódica de errores.
    
    Attributes:
        code: Código de corrección de errores a usar
        error_model: Modelo de errores del sistema
        correction_active: Si la corrección está activa
        measurement_frequency: Frecuencia de mediciones de corrección
    """

    def __init__(self, code: QuantumErrorCorrectionCode, error_model: ErrorModel):
        """
        Inicializa el simulador.
        
        Args:
            code: Código de corrección de errores
            error_model: Modelo de errores del sistema
        """
        self.code = code
        self.error_model = error_model
        self.correction_active = True
        self.measurement_frequency = 1.0

    def generate_lindblad_operators(self) -> List[np.ndarray]:
        """
        Genera operadores de Lindblad basados en el modelo de error.
        
        Los operadores de Lindblad L_k definen la dinámica de error a través
        de la ecuación de Lindblad: dρ/dt = Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
        
        Returns:
            List[np.ndarray]: Lista de operadores de Lindblad
        """
        L_ops = []

        if self.error_model.error_type == ErrorType.BIT_FLIP:
            rate = self.error_model.error_rates.get('gamma', 0.01)
            # Operadores de Lindblad para errores de bit-flip en cada qubit
            for i in range(self.code.n_physical):
                X_i = self._single_qubit_operator('X', i)
                L_ops.append(np.sqrt(rate) * X_i)

        elif self.error_model.error_type == ErrorType.DEPOLARIZING:
            rate = self.error_model.error_rates.get('gamma', 0.01)
            # Canal despolarizante en cada qubit
            for i in range(self.code.n_physical):
                for pauli in ['X', 'Y', 'Z']:
                    P_i = self._single_qubit_operator(pauli, i)
                    L_ops.append(np.sqrt(rate/3) * P_i)

        elif self.error_model.error_type == ErrorType.PHASE_FLIP:
            rate = self.error_model.error_rates.get('gamma', 0.01)
            # Operadores de Lindblad para errores de phase-flip
            for i in range(self.code.n_physical):
                Z_i = self._single_qubit_operator('Z', i)
                L_ops.append(np.sqrt(rate) * Z_i)

        # Se pueden añadir más tipos de error aquí

        return L_ops

    def _single_qubit_operator(self, pauli: str, qubit_index: int) -> np.ndarray:
        """
        Construye operador de Pauli para un qubit específico.
        
        Args:
            pauli: Tipo de operador Pauli ('I', 'X', 'Y', 'Z')
            qubit_index: Índice del qubit (0 a n_physical-1)
            
        Returns:
            np.ndarray: Matriz del operador multi-qubit
        """
        I = np.eye(2)
        pauli_matrices = {
            'I': I,
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }

        op = np.array([[1.0]])  # Inicializar como escalar
        for i in range(self.code.n_physical):
            if i == qubit_index:
                op = np.kron(op, pauli_matrices[pauli])
            else:
                op = np.kron(op, I)

        return op

    def simulate_with_correction(self, 
                               initial_state: np.ndarray,
                               t_span: Tuple[float, float],
                               correction_intervals: List[float]) -> Dict[str, Any]:
        """
        Simula evolución con corrección periódica de errores.
        
        Args:
            initial_state: Estado lógico inicial
            t_span: Tupla (t_inicial, t_final) para la simulación
            correction_intervals: Lista de tiempos para aplicar corrección
            
        Returns:
            Dict[str, Any]: Resultados de la simulación incluyendo:
                - times: Tiempos de la simulación
                - fidelities: Fidelidades en función del tiempo
                - logical_error_prob: Probabilidades de error lógico
                - correction_events: Eventos de corrección
        """
        logging.info("Iniciando simulación...")
        
        # Codificar el estado inicial
        encoded_state = self.code.encode_state(initial_state)
        rho = np.outer(encoded_state, encoded_state.conj())

        # Generar operadores de Lindblad
        L_ops = self.generate_lindblad_operators()

        # Inicializar resultados
        results = {
            'times': [],
            'fidelities': [],
            'logical_error_prob': [],
            'correction_events': []
        }

        # Parámetros de simulación
        current_time = t_span[0]
        num_steps = 1000
        dt = (t_span[1] - t_span[0]) / num_steps
        target_rho = np.outer(encoded_state, encoded_state.conj())

        # Bucle principal de simulación
        for step in range(num_steps):
            # Evolución libre con errores (ecuación de Lindblad)
            rho = self._lindblad_step(rho, L_ops, dt)

            # Verificar si es tiempo de corrección
            if any(abs(current_time - t_corr) < dt/2 for t_corr in correction_intervals):
                if self.correction_active:
                    logging.info(f"Corrección de errores en t={current_time:.2f}")
                    rho, correction_success = self._perform_error_correction(rho)
                    results['correction_events'].append({
                        'time': current_time,
                        'success': correction_success
                    })

            # Calcular métricas
            fidelity = np.real(np.trace(target_rho @ rho))
            logical_error = self._calculate_logical_error_probability(rho)

            # Guardar resultados
            results['times'].append(current_time)
            results['fidelities'].append(fidelity)
            results['logical_error_prob'].append(logical_error)

            current_time += dt

        logging.info("Simulación completada.")
        return results

    def _lindblad_step(self, rho: np.ndarray, L_ops: List[np.ndarray], dt: float) -> np.ndarray:
        """
        Realiza un paso de evolución según la ecuación de Lindblad.
        
        La ecuación de Lindblad es:
        dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
        
        Args:
            rho: Matriz de densidad actual
            L_ops: Lista de operadores de Lindblad
            dt: Paso temporal
            
        Returns:
            np.ndarray: Matriz de densidad evolucionada
        """
        # Hamiltoniano (asumimos H=0 para simplicidad)
        H = np.zeros_like(rho)

        # Término de Lindblad
        L_term = np.zeros_like(rho, dtype=complex)
        for L in L_ops:
            L_dag = L.conj().T
            L_term += (L @ rho @ L_dag - 
                      0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))

        # Integración de Euler
        drho_dt = -1j * (H @ rho - rho @ H) + L_term
        return rho + dt * drho_dt

    def _perform_error_correction(self, rho: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Realiza corrección de errores basada en medición de síndrome.
        
        Args:
            rho: Matriz de densidad actual
            
        Returns:
            Tuple[np.ndarray, bool]: (matriz corregida, éxito de corrección)
        """
        # Medir síndromes
        syndromes = []
        for stabilizer in self.code.get_stabilizers():
            # El valor esperado del estabilizador
            syndrome_val = np.real(np.trace(stabilizer @ rho))
            syndromes.append(1 if syndrome_val > 0 else -1)

        # Decodificar error
        error_location = self._decode_syndrome(syndromes)

        if error_location is not None:
            # Aplicar corrección
            correction_op = self._get_correction_operator(error_location)
            rho_corrected = correction_op @ rho @ correction_op.conj().T
            return rho_corrected, True
        else:
            return rho, False

    def _decode_syndrome(self, syndromes: List[int]) -> Optional[int]:
        """
        Decodifica síndrome para encontrar la ubicación del error.
        
        Args:
            syndromes: Lista de mediciones de síndrome (+1 o -1)
            
        Returns:
            Optional[int]: Ubicación del error o None si no hay error
        """
        if isinstance(self.code, ThreeQubitBitFlipCode):
            # Tabla de lookup para código de 3 qubits
            syndrome_table = {
                (1, 1): None,    # Sin error
                (-1, 1): 0,      # Error en qubit 0
                (-1, -1): 1,     # Error en qubit 1
                (1, -1): 2,      # Error en qubit 2
            }
            return syndrome_table.get(tuple(syndromes))

        elif isinstance(self.code, ShorNineQubitCode):
            # Para el código de Shor, la decodificación es más compleja
            if any(s == -1 for s in syndromes):
                # Simplificación: devolver primera ubicación de error
                return syndromes.index(-1) if -1 in syndromes else 0

        return None

    def _get_correction_operator(self, error_location: int) -> np.ndarray:
        """
        Retorna operador de corrección para la ubicación de error dada.
        
        Args:
            error_location: Índice del qubit con error
            
        Returns:
            np.ndarray: Operador de corrección
        """
        if isinstance(self.code, ThreeQubitBitFlipCode):
            # Aplicar X en la ubicación del error de bit-flip
            return self.code.X[error_location]

        elif isinstance(self.code, ShorNineQubitCode):
            # Para el código de Shor, aplicamos corrección X
            return self._single_qubit_operator('X', error_location)

        return np.eye(self.code.dimension)

    def _calculate_logical_error_probability(self, rho: np.ndarray) -> float:
        """
        Calcula la probabilidad de error lógico.
        
        Args:
            rho: Matriz de densidad actual
            
        Returns:
            float: Probabilidad de error lógico [0, 1]
        """
        logical_projectors = self._get_logical_projectors()

        # Probabilidad de estar en el espacio de código
        code_space_prob = sum(np.real(np.trace(proj @ rho)) 
                            for proj in logical_projectors)

        # El error lógico es la probabilidad de estar fuera del subespacio
        return max(0.0, 1.0 - code_space_prob)

    def _get_logical_projectors(self) -> List[np.ndarray]:
        """
        Retorna proyectores sobre los estados lógicos del código.
        
        Returns:
            List[np.ndarray]: Lista de proyectores lógicos
        """
        if isinstance(self.code, ThreeQubitBitFlipCode):
            # Estados lógicos |0⟩_L y |1⟩_L
            psi_0_L = np.zeros(8)
            psi_0_L[0] = 1/np.sqrt(2)  # |000⟩
            psi_0_L[7] = 1/np.sqrt(2)  # |111⟩

            psi_1_L = np.zeros(8)
            psi_1_L[0] = 1/np.sqrt(2)   # |000⟩
            psi_1_L[7] = -1/np.sqrt(2)  # |111⟩

            return [
                np.outer(psi_0_L, psi_0_L.conj()),
                np.outer(psi_1_L, psi_1_L.conj())
            ]

        elif isinstance(self.code, ShorNineQubitCode):
            # Proyectores para el código de Shor
            psi_0_L = self.code.encode_state(np.array([1, 0]))
            psi_1_L = self.code.encode_state(np.array([0, 1]))
            return [
                np.outer(psi_0_L, psi_0_L.conj()),
                np.outer(psi_1_L, psi_1_L.conj())
            ]

        return []


# ============================================================================
# OPTIMIZACIÓN DE PROTOCOLOS
# ============================================================================

class ProtocolOptimizer:
    """
    Optimizador para protocolos de corrección de errores cuánticos.
    
    Permite optimizar parámetros como frecuencia de corrección,
    umbrales de error, y otros aspectos del protocolo de corrección.
    
    Attributes:
        simulator: Simulador de corrección de errores
    """

    def __init__(self, simulator: QuantumErrorCorrectionSimulator):
        """
        Inicializa el optimizador.
        
        Args:
            simulator: Simulador de corrección de errores a optimizar
        """
        self.simulator = simulator

    def optimize_correction_frequency(self,
                                     initial_state: np.ndarray,
                                     t_span: Tuple[float, float],
                                     frequency_range: Tuple[float, float]) -> Dict[str, Any]:
        """
        Optimiza la frecuencia de corrección para maximizar la fidelidad.
        
        Utiliza optimización numérica para encontrar la frecuencia óptima
        de aplicación de corrección de errores que maximice la fidelidad final.
        
        Args:
            initial_state: Estado lógico inicial para la optimización
            t_span: Intervalo de tiempo (t_inicial, t_final)
            frequency_range: Rango de frecuencias (f_min, f_max) en Hz
            
        Returns:
            Dict[str, Any]: Resultados de la optimización:
                - optimal_frequency: Frecuencia óptima encontrada
                - optimal_fidelity: Fidelidad obtenida con frecuencia óptima
                - optimization_result: Resultado completo del optimizador
        """
        logging.info("Iniciando optimización de frecuencia...")

        def objective(frequency: np.ndarray) -> float:
            """
            Función objetivo para la optimización.
            
            Args:
                frequency: Array con la frecuencia a evaluar
                
            Returns:
                float: Valor negativo de la fidelidad (para minimizar)
            """
            self.simulator.correction_active = True
            freq_val = frequency[0]
            
            # Evitar división por cero
            if freq_val <= 0:
                return 1.0  # Penalizar frecuencias no válidas
                
            intervals = np.arange(t_span[0], t_span[1], 1.0/freq_val)

            try:
                results = self.simulator.simulate_with_correction(
                    initial_state, t_span, intervals.tolist()
                )
                fidelities = np.array(results['fidelities'])
                # Retornar negativo para minimización (queremos maximizar fidelidad)
                return -fidelities[-1]
            except Exception as e:
                logging.warning(f"Error en evaluación de frecuencia {freq_val}: {e}")
                return 1.0  # Penalizar en caso de error

        # Optimización usando L-BFGS-B
        result = minimize(
            objective,
            x0=[(frequency_range[0] + frequency_range[1])/2],
            bounds=[frequency_range],
            method='L-BFGS-B'
        )

        optimal_frequency = result.x[0]
        optimal_fidelity = -result.fun

        logging.info(f"Optimización completada. Frecuencia óptima: {optimal_frequency:.3f} Hz")
        
        return {
            'optimal_frequency': optimal_frequency,
            'optimal_fidelity': optimal_fidelity,
            'optimization_result': result
        }


# ============================================================================
# ANÁLISIS Y BENCHMARKING
# ============================================================================

def benchmark_codes(codes: List[QuantumErrorCorrectionCode],
                    error_rates: List[float],
                    initial_state: np.ndarray) -> Dict[str, Any]:
    """
    Realiza benchmark comparativo de diferentes códigos de corrección.
    
    Compara el rendimiento de múltiples códigos bajo diferentes tasas de error,
    evaluando métricas como fidelidad final y factor de mejora.
    
    Args:
        codes: Lista de códigos de corrección a comparar
        error_rates: Lista de tasas de error a evaluar
        initial_state: Estado lógico inicial para las pruebas
        
    Returns:
        Dict[str, Any]: Resultados del benchmark organizados por código y tasa
    """
    logging.info("Iniciando benchmark de códigos...")
    results = {}

    for code in codes:
        logging.info(f"Evaluando código: {code.name}")
        code_results = {}

        for rate in error_rates:
            logging.info(f"  Tasa de error: {rate}")
            
            # Crear modelo de error
            error_model = ErrorModel(
                error_type=ErrorType.BIT_FLIP,
                error_rates={'gamma': rate}
            )

            # Crear simulador
            simulator = QuantumErrorCorrectionSimulator(code, error_model)

            # Parámetros de simulación
            t_span = (0, min(10.0, 5.0 / rate))  # Tiempo adaptativo
            correction_intervals = np.arange(0, t_span[1], 
                                           min(1.0, 0.5/rate)).tolist()

            try:
                # Simular con corrección
                simulator.correction_active = True
                with_correction = simulator.simulate_with_correction(
                    initial_state, t_span, correction_intervals
                )

                # Simular sin corrección
                simulator.correction_active = False
                without_correction = simulator.simulate_with_correction(
                    initial_state, t_span, []
                )

                # Calcular factor de mejora
                improvement_factor = (
                    with_correction['fidelities'][-1] /
                    max(without_correction['fidelities'][-1], 1e-10)
                )

                code_results[rate] = {
                    'with_correction': with_correction,
                    'without_correction': without_correction,
                    'improvement_factor': improvement_factor,
                    'final_fidelity_with': with_correction['fidelities'][-1],
                    'final_fidelity_without': without_correction['fidelities'][-1]
                }
                
            except Exception as e:
                logging.error(f"Error en benchmark para {code.name} con tasa {rate}: {e}")
                code_results[rate] = {
                    'error': str(e),
                    'improvement_factor': 0.0
                }

        results[code.name] = code_results

    logging.info("Benchmark completado.")
    return results


def analyze_threshold_behavior(code: QuantumErrorCorrectionCode,
                             error_rates: np.ndarray,
                             initial_state: np.ndarray,
                             t_final: float = 10.0) -> Dict[str, Any]:
    """
    Analiza el comportamiento de umbral del código de corrección.
    
    Estudia cómo varía el rendimiento del código cerca del umbral de error,
    punto crítico donde la corrección deja de ser efectiva.
    
    Args:
        code: Código de corrección a analizar
        error_rates: Array de tasas de error a evaluar
        initial_state: Estado inicial para el análisis
        t_final: Tiempo final de simulación
        
    Returns:
        Dict[str, Any]: Análisis del comportamiento de umbral
    """
    logging.info(f"Analizando comportamiento de umbral para {code.name}")
    
    threshold_results = {
        'error_rates': error_rates,
        'logical_error_rates': [],
        'fidelities': [],
        'improvement_factors': []
    }
    
    for rate in error_rates:
        error_model = ErrorModel(
            error_type=ErrorType.BIT_FLIP,
            error_rates={'gamma': rate}
        )
        
        simulator = QuantumErrorCorrectionSimulator(code, error_model)
        t_span = (0, t_final)
        correction_intervals = np.arange(0, t_final, 0.5/max(rate, 0.01)).tolist()
        
        try:
            # Simular con corrección
            results = simulator.simulate_with_correction(
                initial_state, t_span, correction_intervals
            )
            
            logical_error_rate = results['logical_error_prob'][-1]
            final_fidelity = results['fidelities'][-1]
            
            # Simular sin corrección para comparar
            simulator.correction_active = False
            no_correction = simulator.simulate_with_correction(
                initial_state, t_span, []
            )
            
            improvement = final_fidelity / max(no_correction['fidelities'][-1], 1e-10)
            
            threshold_results['logical_error_rates'].append(logical_error_rate)
            threshold_results['fidelities'].append(final_fidelity)
            threshold_results['improvement_factors'].append(improvement)
            
        except Exception as e:
            logging.warning(f"Error en tasa {rate}: {e}")
            threshold_results['logical_error_rates'].append(1.0)
            threshold_results['fidelities'].append(0.0)
            threshold_results['improvement_factors'].append(0.0)
    
    return threshold_results


# ============================================================================
# UTILIDADES DE VISUALIZACIÓN
# ============================================================================

def plot_benchmark_results(benchmark_results: Dict[str, Any], 
                          save_path: Optional[str] = None) -> go.Figure:
    """
    Crea visualización interactiva de los resultados de benchmark.
    
    Args:
        benchmark_results: Resultados del benchmark
        save_path: Ruta para guardar la figura (opcional)
        
    Returns:
        go.Figure: Figura de Plotly con los resultados
    """
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (code_name, code_results) in enumerate(benchmark_results.items()):
        error_rates = []
        improvement_factors = []
        
        for rate, results in code_results.items():
            if isinstance(results, dict) and 'improvement_factor' in results:
                error_rates.append(rate)
                improvement_factors.append(results['improvement_factor'])
        
        if error_rates:  # Solo plotear si hay datos válidos
            fig.add_trace(go.Scatter(
                x=error_rates,
                y=improvement_factors,
                mode='lines+markers',
                name=code_name,
                line=dict(color=colors[i % len(colors)]),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Tasa de Error: %{x:.4f}<br>' +
                            'Factor de Mejora: %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Benchmark de Códigos de Corrección de Errores',
        xaxis_title='Tasa de Error',
        yaxis_title='Factor de Mejora (Fidelidad con/sin corrección)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def main_example() -> None:
    """
    Ejemplo principal demostrando el uso completo del framework.
    
    Muestra cómo usar todas las funcionalidades principales:
    - Creación de códigos y simulación
    - Comparación con/sin corrección  
    - Optimización de parámetros
    - Benchmark y análisis
    """
    print("=== FRAMEWORK DE CORRECCIÓN DE ERRORES CUÁNTICOS ===\n")

    # 1. Crear códigos de corrección
    bit_flip_code = ThreeQubitBitFlipCode()
    shor_code = ShorNineQubitCode()  # Ahora disponible sin warnings

    # 2. Definir modelo de error
    error_model = ErrorModel(
        error_type=ErrorType.BIT_FLIP,
        error_rates={'gamma': 0.05}  # 5% de tasa de error
    )

    # 3. Estado inicial lógico |+⟩ = (|0⟩ + |1⟩)/√2
    initial_logical = np.array([1, 1]) / np.sqrt(2)

    # 4. Crear simulador
    simulator = QuantumErrorCorrectionSimulator(bit_flip_code, error_model)

    # 5. Simular evolución temporal
    t_span = (0, 10.0)
    correction_intervals = np.arange(0, 10, 1.0).tolist()

    print("Ejecutando simulación con corrección...")
    results_with_correction = simulator.simulate_with_correction(
        initial_logical, t_span, correction_intervals
    )

    print("Ejecutando simulación sin corrección...")
    simulator.correction_active = False
    results_without_correction = simulator.simulate_with_correction(
        initial_logical, t_span, []
    )

    # 6. Mostrar resultados
    print(f"\nResultados de la simulación:")
    print(f"Fidelidad inicial: {results_without_correction['fidelities'][0]:.4f}")
    print(f"Fidelidad final (sin corrección): {results_without_correction['fidelities'][-1]:.4f}")
    print(f"Fidelidad final (con corrección): {results_with_correction['fidelities'][-1]:.4f}")
    
    improvement = (results_with_correction['fidelities'][-1] / 
                  results_without_correction['fidelities'][-1])
    print(f"Factor de mejora: {improvement:.2f}x")

    # 7. Visualización interactiva
    print("\nCreando visualización...")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_without_correction['times'],
        y=results_without_correction['fidelities'],
        mode='lines',
        name='Sin Corrección',
        line=dict(dash='dash', color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=results_with_correction['times'],
        y=results_with_correction['fidelities'],
        mode='lines',
        name='Con Corrección',
        line=dict(color='blue', width=2)
    ))

    # Marcar eventos de corrección
    correction_times = [e['time'] for e in results_with_correction['correction_events']]
    for i, t_corr in enumerate(correction_times):
        fig.add_vline(
            x=t_corr, 
            line_dash="dot", 
            line_color="green", 
            annotation_text=f"Corrección {i+1}",
            annotation_position="top"
        )

    fig.update_layout(
        title='Evolución de Fidelidad: Efecto de la Corrección de Errores',
        xaxis_title='Tiempo',
        yaxis_title='Fidelidad del Estado Cuántico',
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )

    fig.show()

    # 8. Optimización de protocolo
    print("\n--- OPTIMIZACIÓN DE FRECUENCIA DE CORRECCIÓN ---")
    simulator.correction_active = True  # Reactivar corrección
    optimizer = ProtocolOptimizer(simulator)
    
    optimization_result = optimizer.optimize_correction_frequency(
        initial_logical, t_span, (0.1, 3.0)
    )

    print(f"Frecuencia óptima: {optimization_result['optimal_frequency']:.3f} Hz")
    print(f"Fidelidad óptima: {optimization_result['optimal_fidelity']:.4f}")

    # 9. Benchmark de códigos
    print("\n--- BENCHMARK COMPARATIVO ---")
    codes_to_compare = [bit_flip_code]  # Agregar shor_code si se desea
    error_rates = [0.01, 0.05, 0.1, 0.15]
    
    benchmark_results = benchmark_codes(codes_to_compare, error_rates, initial_logical)
    
    # Mostrar resultados del benchmark
    for code_name, results in benchmark_results.items():
        print(f"\nCódigo: {code_name}")
        for rate, result in results.items():
            if isinstance(result, dict) and 'improvement_factor' in result:
                print(f"  Tasa {rate}: Mejora {result['improvement_factor']:.2f}x")

    print("\n=== SIMULACIÓN COMPLETADA ===")


if __name__ == "__main__":
    main_example()
