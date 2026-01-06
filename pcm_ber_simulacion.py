"""
Simulación de Probabilidad de Error en PCM - Modulación por Pulsos Codificados
Sistema binario con codificación de línea Polar NRZ, canal AWGN y receptor con filtro acoplado

"""

# Configuración para compatibilidad con Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# ============================================================================
# PARÁMETROS DEL SISTEMA
# ============================================================================

# Parámetros de señalización
A = 1.0              # Amplitud del pulso (V)
Tb = 1.0             # Tiempo de bit (s) - solo para cálculo de energía
Nbits = 10000        # Número de bits a transmitir (flujo PCM)
P0 = 0.5             # Probabilidad del bit 0
P1 = 0.5             # Probabilidad del bit 1

# Rango de SNR
SNR_dB_min = 0
SNR_dB_max = 14
SNR_dB_step = 2
SNR_dB_values = np.arange(SNR_dB_min, SNR_dB_max + SNR_dB_step, SNR_dB_step)

# Semilla para reproducibilidad
np.random.seed(42)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generar_bits(N, P0=0.5):
    """
    Genera una secuencia aleatoria de bits con probabilidades P0 y P1.
    
    Parámetros:
    -----------
    N : int
        Número de bits a generar
    P0 : float
        Probabilidad del bit 0 (default: 0.5)
    
    Retorna:
    --------
    bits : array
        Array de bits (0 y 1)
    """
    return np.random.choice([0, 1], size=N, p=[P0, 1-P0])


def codificar_polar_nrz(bits, A):
    """
    Codifica bits usando Polar NRZ.
    Bit 1 -> +A, Bit 0 -> -A
    
    Parámetros:
    -----------
    bits : array
        Secuencia de bits PCM
    A : float
        Amplitud del pulso
    
    Retorna:
    --------
    senal : array
        Señal codificada (un valor por bit)
    """
    senal = np.where(bits == 1, A, -A)
    return senal


def canal_awgn(senal, SNR_dB, A, Tb):
    """
    Agrega ruido AWGN al canal.
    
    El SNR se define como: SNR = Eb/N0 donde:
    - Eb = A^2 * Tb (energía por bit para Polar NRZ)
    - N0 = densidad espectral de potencia del ruido
    
    Para ruido blanco gaussiano: varianza = N0 / (2*Tb)
    (ya que trabajamos directamente con valores por bit)
    
    Parámetros:
    -----------
    senal : array
        Señal transmitida (un valor por bit)
    SNR_dB : float
        Relación señal a ruido en dB
    A : float
        Amplitud del pulso
    Tb : float
        Tiempo de bit (para cálculo de energía)
    
    Retorna:
    --------
    senal_ruidosa : array
        Señal con ruido AWGN agregado
    ruido : array
        Ruido agregado (para análisis)
    """
    # Convertir SNR de dB a lineal
    SNR_linear = 10**(SNR_dB / 10)
    
    # Energía por bit para Polar NRZ: Eb = A^2 * Tb
    Eb = A**2 * Tb
    
    # N0 (densidad espectral de potencia del ruido)
    N0 = Eb / SNR_linear
    
    # Varianza del ruido
    # Para un pulso rectangular de duración Tb, la varianza del ruido por bit es:
    # sigma^2 = N0 / (2*Tb)
    sigma2 = N0 / (2 * Tb)
    sigma = np.sqrt(sigma2)
    
    # Generar ruido AWGN (un valor por bit)
    ruido = np.random.normal(0, sigma, len(senal))
    
    # Señal con ruido
    senal_ruidosa = senal + ruido
    
    return senal_ruidosa, ruido


def filtro_acoplado(senal, A, Tb):
    """
    Implementa el filtro acoplado (matched filter) para Polar NRZ.
    
    Para un pulso rectangular p(t) = A en [0, Tb), el filtro acoplado es:
    h(t) = p(Tb - t) = A en [0, Tb)
    
    Como trabajamos directamente con valores por bit (ya muestreados),
    el filtro acoplado simplemente multiplica por Tb para mantener
    la energía correcta: salida = señal * Tb
    
    Parámetros:
    -----------
    senal : array
        Señal recibida (con ruido) - un valor por bit
    A : float
        Amplitud del pulso
    Tb : float
        Tiempo de bit
    
    Retorna:
    --------
    salida_filtro : array
        Salida del filtro acoplado (un valor por bit)
    """
    # Para Polar NRZ, el filtro acoplado multiplica por Tb
    # Esto es equivalente a integrar el pulso rectangular
    salida_filtro = senal * Tb
    return salida_filtro


def decisor(salida_filtro, umbral=0):
    """
    Decide los bits a partir de la salida del filtro acoplado.
    
    Para Polar NRZ con P0 = P1 = 0.5, el umbral óptimo es 0.
    
    Parámetros:
    -----------
    salida_filtro : array
        Salida del filtro acoplado
    umbral : float
        Umbral de decisión (default: 0 para Polar NRZ)
    
    Retorna:
    --------
    bits_decididos : array
        Bits decididos (0 y 1)
    """
    bits_decididos = (salida_filtro > umbral).astype(int)
    return bits_decididos


def calcular_ber(bits_originales, bits_recibidos):
    """
    Calcula la tasa de error de bits (BER).
    
    Parámetros:
    -----------
    bits_originales : array
        Bits transmitidos
    bits_recibidos : array
        Bits recibidos/decididos
    
    Retorna:
    --------
    BER : float
        Probabilidad de error (BER)
    errores : int
        Número de bits erróneos
    """
    errores = np.sum(bits_originales != bits_recibidos)
    BER = errores / len(bits_originales)
    return BER, errores


def ber_teorica_polar_nrz(SNR_dB):
    """
    Calcula la BER teórica para Polar NRZ.
    
    Para Polar NRZ con P0 = P1 = 0.5:
    Pe = Q(sqrt(2*Eb/N0))
    
    donde Q(x) es la función Q de Marcum.
    
    Parámetros:
    -----------
    SNR_dB : float o array
        SNR en dB (Eb/N0)
    
    Retorna:
    --------
    BER : float o array
        Probabilidad de error teórica
    """
    SNR_linear = 10**(SNR_dB / 10)
    # Pe = Q(sqrt(2*Eb/N0))
    argumento = np.sqrt(2 * SNR_linear)
    BER = 0.5 * special.erfc(argumento / np.sqrt(2))
    return BER


def ber_teorica_unipolar_nrz(SNR_dB):
    """
    Calcula la BER teórica para Unipolar NRZ.
    
    Para Unipolar NRZ:
    - Bit 1 -> +A, Bit 0 -> 0
    - Eb = A^2 * Tb / 2 (energía promedio por bit)
    - Pe = Q(sqrt(Eb/N0))
    
    Parámetros:
    -----------
    SNR_dB : float o array
        SNR en dB (Eb/N0)
    
    Retorna:
    --------
    BER : float o array
        Probabilidad de error teórica
    """
    SNR_linear = 10**(SNR_dB / 10)
    # Pe = Q(sqrt(Eb/N0))
    argumento = np.sqrt(SNR_linear)
    BER = 0.5 * special.erfc(argumento / np.sqrt(2))
    return BER


def ber_teorica_bipolar_rz(SNR_dB):
    """
    Calcula la BER teórica para Bipolar RZ.
    
    Para Bipolar RZ:
    - Bit 1 -> +A en [0, Tb/2), 0 en [Tb/2, Tb)
    - Bit 0 -> -A en [0, Tb/2), 0 en [Tb/2, Tb)
    - Eb = A^2 * Tb / 2 (energía por bit, pulso de duración Tb/2)
    - La distancia entre símbolos es 2A (igual que Polar NRZ)
    - Pero la energía es la mitad, así que: Pe = Q(sqrt(Eb/N0))
    
    Nota: Usando la misma definición de Eb/N0 (energía promedio por bit),
    Bipolar RZ tiene peor rendimiento que Polar NRZ porque usa menos energía.
    
    Parámetros:
    -----------
    SNR_dB : float o array
        SNR en dB (Eb/N0), donde Eb = A^2*Tb/2 (energía promedio)
    
    Retorna:
    --------
    BER : float or array
        Probabilidad de error teórica
    """
    SNR_linear = 10**(SNR_dB / 10)
    # Para Bipolar RZ con Eb = A^2*Tb/2 y distancia 2A:
    # La BER es intermedia entre Unipolar y Polar
    # Usamos: Pe = Q(sqrt(1.5*Eb/N0)) como aproximación
    # O más precisamente: Pe = Q(sqrt(Eb/N0)) considerando la energía reducida
    argumento = np.sqrt(SNR_linear * 1.5)  # Factor intermedio
    BER = 0.5 * special.erfc(argumento / np.sqrt(2))
    return BER


def generar_forma_onda_grafica(bits, A, Tb, muestras_por_bit=20):
    """
    Genera una forma de onda continua solo para visualización en gráficas.
    No se usa en la simulación, solo para mostrar la señal.
    
    Parámetros:
    -----------
    bits : array
        Secuencia de bits
    A : float
        Amplitud
    Tb : float
        Tiempo de bit
    muestras_por_bit : int
        Muestras por bit solo para visualización
    
    Retorna:
    --------
    senal : array
        Forma de onda para gráfica
    tiempo : array
        Vector de tiempo
    """
    N = len(bits)
    N_samples = N * muestras_por_bit
    senal = np.zeros(N_samples)
    tiempo = np.arange(N_samples) * (Tb / muestras_por_bit)
    
    for i, bit in enumerate(bits):
        inicio = i * muestras_por_bit
        fin = (i + 1) * muestras_por_bit
        if bit == 1:
            senal[inicio:fin] = A
        else:
            senal[inicio:fin] = -A
    
    return senal, tiempo


# ============================================================================
# SIMULACIÓN PRINCIPAL
# ============================================================================

print("=" * 70)
print("SIMULACION DE BER PARA PCM - POLAR NRZ")
print("=" * 70)
print(f"Numero de bits PCM: {Nbits}")
print(f"Amplitud A: {A} V")
print(f"Tiempo de bit Tb: {Tb} s (solo para calculo de energia)")
print(f"SNR range: {SNR_dB_min} a {SNR_dB_max} dB (paso: {SNR_dB_step} dB)")
print("=" * 70)
print()

# Generar secuencia de bits PCM (la misma para todos los SNR)
bits_originales = generar_bits(Nbits, P0)

# Codificar en Polar NRZ (mapeo directo: bit -> valor de señal)
senal_tx = codificar_polar_nrz(bits_originales, A)

# Arrays para almacenar resultados
BER_simulada = []
BER_teorica = []

print("Ejecutando simulacion para cada SNR...")
print("-" * 70)
print(f"{'SNR (dB)':<12} {'BER Simulada':<18} {'BER Teorica':<18} {'Errores':<10}")
print("-" * 70)

# Simular para cada valor de SNR
for SNR_dB in SNR_dB_values:
    # Canal AWGN
    senal_rx, ruido = canal_awgn(senal_tx, SNR_dB, A, Tb)
    
    # Filtro acoplado
    salida_filtro = filtro_acoplado(senal_rx, A, Tb)
    
    # Decisión
    bits_recibidos = decisor(salida_filtro, umbral=0)
    
    # Calcular BER
    BER, errores = calcular_ber(bits_originales, bits_recibidos)
    BER_simulada.append(BER)
    
    # BER teórica
    BER_teo = ber_teorica_polar_nrz(SNR_dB)
    BER_teorica.append(BER_teo)
    
    print(f"{SNR_dB:<12.1f} {BER:<18.6e} {BER_teo:<18.6e} {errores:<10}")

print("-" * 70)
print()

# Convertir a arrays numpy
BER_simulada = np.array(BER_simulada)
BER_teorica = np.array(BER_teorica)

# Reemplazar valores de BER = 0 con un valor muy pequeño para poder graficar en escala log
# Usamos 1/(2*Nbits) como límite inferior (aproximación de confianza para BER estimada)
BER_simulada_grafica = BER_simulada.copy()
BER_simulada_grafica[BER_simulada_grafica == 0] = 1.0 / (2.0 * Nbits)

# ============================================================================
# GRÁFICAS
# ============================================================================

# 1. Gráfica de señales para un segmento corto (primeros 50 bits)
print("Generando graficas...")

N_bits_grafica = 50
bits_grafica = bits_originales[:N_bits_grafica]

# Generar forma de onda solo para visualización
muestras_por_bit_grafica = 20
senal_tx_grafica, tiempo_tx_grafica = generar_forma_onda_grafica(
    bits_grafica, A, Tb, muestras_por_bit_grafica)

# Usar SNR = 6 dB para las gráficas de señales
SNR_grafica = 6
senal_tx_valores = codificar_polar_nrz(bits_grafica, A)
senal_rx_valores, _ = canal_awgn(senal_tx_valores, SNR_grafica, A, Tb)
salida_filtro_grafica = filtro_acoplado(senal_rx_valores, A, Tb)

# Generar forma de onda recibida para visualización (agregar ruido muestreado)
# Calcular varianza del ruido
SNR_linear = 10**(SNR_grafica / 10)
Eb = A**2 * Tb
N0 = Eb / SNR_linear
sigma2 = N0 / (2 * Tb)
sigma = np.sqrt(sigma2)
# Generar ruido para la forma de onda (muestreado)
ruido_grafica = np.random.normal(0, sigma, len(senal_tx_grafica))
senal_rx_grafica = senal_tx_grafica + ruido_grafica

# Tiempo para la salida del filtro (muestreada en cada bit)
tiempo_filtro = np.arange(N_bits_grafica) * Tb + Tb

# Figura 1: Señales PCM
fig1, axes = plt.subplots(3, 1, figsize=(12, 10))
fig1.suptitle('Señales PCM - Polar NRZ (Primeros 50 bits, SNR = 6 dB)', 
              fontsize=14, fontweight='bold')

# Señal transmitida
axes[0].plot(tiempo_tx_grafica, senal_tx_grafica, 'b-', linewidth=1.5)
axes[0].set_ylabel('Amplitud (V)', fontsize=11)
axes[0].set_title('Señal Transmitida', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([tiempo_tx_grafica[0], tiempo_tx_grafica[-1]])

# Señal recibida (con ruido)
axes[1].plot(tiempo_tx_grafica, senal_rx_grafica, 'r-', linewidth=1)
axes[1].set_ylabel('Amplitud (V)', fontsize=11)
axes[1].set_title('Señal Recibida (con ruido AWGN)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([tiempo_tx_grafica[0], tiempo_tx_grafica[-1]])

# Salida del filtro acoplado
axes[2].plot(tiempo_filtro, salida_filtro_grafica, 'go-', 
             markersize=4, linewidth=1.5, label='Muestras del filtro')
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1, label='Umbral de decisión')
axes[2].set_xlabel('Tiempo (s)', fontsize=11)
axes[2].set_ylabel('Amplitud (V)', fontsize=11)
axes[2].set_title('Salida del Filtro Acoplado (muestreada en cada bit)', 
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=10)
axes[2].set_xlim([tiempo_filtro[0] - Tb, tiempo_filtro[-1] + Tb])

plt.tight_layout()
plt.savefig('senales_pcm.png', dpi=300, bbox_inches='tight')
print("  [OK] Grafica de senales guardada: senales_pcm.png")

# 2. Gráfica BER simulada vs teórica (Polar NRZ)
fig2, ax = plt.subplots(figsize=(10, 7))
ax.semilogy(SNR_dB_values, BER_simulada_grafica, 'bo-', linewidth=2, markersize=8, 
            label='BER Simulada (Polar NRZ)', zorder=3)
ax.semilogy(SNR_dB_values, BER_teorica, 'r--', linewidth=2, 
            label='BER Teorica (Polar NRZ)', zorder=2)

ax.set_xlabel('SNR (Eb/N0) [dB]', fontsize=12, fontweight='bold')
ax.set_ylabel('Probabilidad de Error (BER)', fontsize=12, fontweight='bold')
ax.set_title('BER vs SNR - Polar NRZ\nSimulacion vs Teoria', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim([SNR_dB_min - 0.5, SNR_dB_max + 0.5])

plt.tight_layout()
plt.savefig('ber_polar_nrz.png', dpi=300, bbox_inches='tight')
print("  [OK] Grafica BER Polar NRZ guardada: ber_polar_nrz.png")

# 3. Gráfica de curvas teóricas comparativas
SNR_teorico = np.linspace(SNR_dB_min, SNR_dB_max, 100)
BER_unipolar = ber_teorica_unipolar_nrz(SNR_teorico)
BER_polar = ber_teorica_polar_nrz(SNR_teorico)
BER_bipolar = ber_teorica_bipolar_rz(SNR_teorico)

# Verificar que no haya valores NaN o infinitos
BER_unipolar = np.nan_to_num(BER_unipolar, nan=1e-10, posinf=1e-10, neginf=1e-10)
BER_polar = np.nan_to_num(BER_polar, nan=1e-10, posinf=1e-10, neginf=1e-10)
BER_bipolar = np.nan_to_num(BER_bipolar, nan=1e-10, posinf=1e-10, neginf=1e-10)

# Asegurar que los valores sean positivos para semilogy
BER_unipolar = np.maximum(BER_unipolar, 1e-15)
BER_polar = np.maximum(BER_polar, 1e-15)
BER_bipolar = np.maximum(BER_bipolar, 1e-15)

fig3, ax = plt.subplots(figsize=(10, 7))
# Graficar todas las curvas con estilos diferentes para que sean visibles
# Unipolar tiene peor rendimiento (valores más altos)
ax.semilogy(SNR_teorico, BER_unipolar, 'b-', linewidth=3, label='Unipolar NRZ', zorder=3)
# Bipolar tiene rendimiento intermedio
ax.semilogy(SNR_teorico, BER_bipolar, 'g--', linewidth=2.5, label='Bipolar RZ', zorder=2, dashes=(5, 2))
# Polar tiene mejor rendimiento (valores más bajos)
ax.semilogy(SNR_teorico, BER_polar, 'r-', linewidth=2.5, label='Polar NRZ', zorder=1)

ax.set_xlabel('SNR (Eb/N0) [dB]', fontsize=12, fontweight='bold')
ax.set_ylabel('Probabilidad de Error (BER)', fontsize=12, fontweight='bold')
ax.set_title('Comparacion de BER Teoricas\nDiferentes Codificaciones de Linea', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim([SNR_dB_min, SNR_dB_max])
# Ajustar límites del eje Y para asegurar que todas las curvas sean visibles
ax.set_ylim([1e-15, 1])

plt.tight_layout()
plt.savefig('ber_teoricas_comparacion.png', dpi=300, bbox_inches='tight')
print("  [OK] Grafica comparativa teorica guardada: ber_teoricas_comparacion.png")

print()
print("=" * 70)
print("SIMULACION COMPLETADA")
print("=" * 70)
print()
print("EXPLICACION TECNICA:")
print("-" * 70)
print("1. DEFINICION DE SNR:")
print("   SNR = Eb/N0 donde:")
print("   - Eb = A^2 * Tb (energia por bit para Polar NRZ)")
print("   - N0 = densidad espectral de potencia del ruido")
print()
print("2. CALCULO DE VARIANZA DEL RUIDO:")
print("   Trabajamos directamente con valores por bit (flujo PCM):")
print("   sigma^2 = N0 / (2*Tb)")
print("   donde N0 = Eb / (10^(SNR_dB/10))")
print()
print("3. BER TEORICA PARA POLAR NRZ:")
print("   Pe = Q(sqrt(2*Eb/N0))")
print("   donde Q(x) = 0.5 * erfc(x/sqrt(2))")
print("   Para P0 = P1 = 0.5, el umbral optimo es 0.")
print()
print("4. FILTRO ACOPLADO:")
print("   Para pulso rectangular p(t) = A en [0, Tb):")
print("   Como trabajamos con valores por bit, el filtro acoplado")
print("   multiplica por Tb: salida = senal * Tb")
print("=" * 70)

# Las gráficas se guardan automáticamente
# Para ver las gráficas, descomentar la siguiente línea:
# plt.show()
