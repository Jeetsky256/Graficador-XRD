import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks

from scipy.optimize import curve_fit

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = diags([w], [0], shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1-p) * (y < z)
    return z
def calculate_fwhm_gaussian(x, y, peak_idx):
    # Establece un rango alrededor del pico para el ajuste
    range_left = max(peak_idx - 35, 0)
    range_right = min(peak_idx + 35, len(x) - 1)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]

    # Realiza el ajuste gaussiano
    popt, pcov = curve_fit(gaussian, x[range_left:range_right], y[range_left:range_right], p0=[peak_y, peak_x, 1])
    amp, cen, wid = popt

    # Calcula el FWHM
    fwhm = 2.355 * abs(wid)  # 2.355 * sigma para una gaussiana
    return fwhm, cen , range_left , range_right , popt

def calculate_fwhm(x, y, peak_idx):
    peak = y[peak_idx]
    y_min = np.amin(y)
    y_prom = np.mean(y)
    half_max = (peak - y_min) / 2.0
    print(y_min, y_prom)
    # Buscar el primer cruce por la izquierda
    try:
        left_idx = np.where(y[:peak_idx] <= half_max)[0][-1]
    except IndexError:
        left_idx = peak_idx
    # Buscar el primer cruce por la derecha
    try:
        right_idx = peak_idx + np.where(y[peak_idx:] <= half_max)[0][0]
    except IndexError:
        right_idx = peak_idx

    fwhm = x[right_idx] - x[left_idx]
    return fwhm, x[peak_idx], x[left_idx], x[right_idx]
# Ruta de la carpeta de entrada y salida
input_dir = 'xrd'
output_dir = 'xrdn'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fwhm_output_path = os.path.join(output_dir, 'fwhm_results.txt')
# Procesar cada archivo en la carpeta de entrada
for filename in os.listdir(input_dir):
    full_path = os.path.join(input_dir, filename)
    if os.path.isfile(full_path):
        # Leer los datos del archivo
        data = np.genfromtxt(full_path, delimiter=',', skip_header=142)

        # Separar los datos en dos columnas
        x = data[:, 0]
        y = data[:, 1]

        # Aplicar la corrección de baseline usando ALS
        lam = 1e5
        p = 0.01
        baseline = baseline_als(y, lam, p)

        # Restar el baseline de los datos originales para corregirlos
        y_corrected = y - baseline

        # Suavizar la curva corregida
        window_length = 21
        polyorder = 2
        y_smoothed = savgol_filter(y_corrected, window_length, polyorder)
        coefficients = np.polyfit(x, y_smoothed, 2)
        polynomial = np.poly1d(coefficients)

        # Calcular la línea de tendencia
        coefficients = np.polyfit(x, y_smoothed, 2)
        polynomial = np.poly1d(coefficients)
        trendline = polynomial(x)
        residuals = y_smoothed -5
        # Detectar picos en la curva suavizada
        peaks, _ = find_peaks(y_smoothed, prominence=101)  # Ajustar parámetros según sea necesario

        peak_heights = y_smoothed[peaks]
        total_peak_height = np.sum(peak_heights)

        # Guardar los datos corregidos en un nuevo archivo en la carpeta de salida
        output_path = os.path.join(output_dir, filename)
        np.savetxt(output_path, np.column_stack((x, y_smoothed)),fmt=['%.8f', '%d'])
        with open(fwhm_output_path, 'a') as f:
            f.write(f'{filename}\n')
            f.write('Máximo | FWHM | Cristalito (\AA) | Distancia interplanar (\AA) | Parámetros de red (\AA) | Intensidad Relativa|\n')
            f.write('-----------------------------------------------------------\n')
            for peak in peaks:
                fwhmdegree, angle , range_left , range_right , popt = calculate_fwhm_gaussian(x, residuals, peak)
               # fwhm, angle, left, right = calculate_fwhm(x, y_smoothed, peak)
                rad = (angle/2) * np.pi /180
                fwhm = fwhmdegree * np.pi /180
                b = (0.9*1.5418)/(fwhm * np.cos(rad)) ## cristalito
                d = 1.5418/(2*np.sin(rad)) #distancia interplanar
                a = 31.769 - angle
                c = 34.421 - angle
                fcc = 38 - angle
                val = 0
                relative_intensity = y_smoothed[peak] / total_peak_height # Intensidad relativa
                if abs(a) < 1:
                    val = d * (4/3)**(1/2)
                    f.write(f'{angle:.3f} | {fwhm:.3f} | {b:.3f} | {d:.3f} | a = {val:.3f} | {relative_intensity:.2f} \n')
                elif abs(c) < 1:
                    val = 2*d
                    f.write(f'{angle:.3f} | {fwhm:.3f} | {b:.3f} | {d:.3f} | c = {val:.3f} | {relative_intensity:.2f} \n')
                elif abs(fcc) < 1 :
                    val = 3**(0.5)*d
                    f.write(f'{angle:.3f} | {fwhm:.3f} | {b:.3f} | {d:.3f} | a(Ag) = {val:.3f} | {relative_intensity:.2f} \n')
                else:
                    f.write(f'{angle:.3f} | {fwhm:.3f} | {b:.3f} | {d:.3f} |            | {relative_intensity:.2f} \n')
            
            f.write('-----------------------------------------------------------\n')
    """# Opcional: crear una gráfica para cada archivo procesado
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Espectro Original')
        plt.plot(x, y_corrected, label='Espectro Corregido')
        plt.plot(x, y_smoothed, label='Espectro Suavizado', linestyle='-')
        plt.plot(x, baseline, label='Baseline', linestyle='--')

        # Marca cada pico con una línea vertical gris
        for peak in peaks:
            fwhm, angle , range_left , range_right , popt = calculate_fwhm_gaussian(x, residuals, peak)
            plt.text(x[peak], y_smoothed[peak]+50, f"{x[peak]:.2f}", ha='center', va='bottom')
            plt.axvline(x=x[peak], color='grey', linestyle='--', linewidth=1)
            plt.plot(x[range_left:range_right], gaussian(x[range_left:range_right], *popt), 'r-', label='Fit: Amp=%5.3f, Cen=%5.3f, Wid=%5.3f' % tuple(popt))

        plt.xlabel(r"$2\theta$")
        plt.ylabel('Intensidad')
        plt.grid(False)
        plt.gca().set_yticklabels([])
        plt.title(f'Corrección y Suavizado del Baseline del Espectro ({filename})')
        plt.legend()


        
        # Guardar la gráfica en el directorio de salida
        graph_output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        plt.savefig(graph_output_path)
        
        #plt.show() #Mostrar gráfica
        plt.close() """
