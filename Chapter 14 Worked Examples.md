Here are the worked examples for Chapter 14, designed to bridge the gap between the main text and the exercises. These examples provide detailed implementations of efficient algorithms for time series analysis, preparing students to tackle computational challenges with the practical wisdom of Feynman, Gelman, Jaynes, and Murphy.

### Worked Example 1: Fast Fourier Transform Optimization and Benchmarking
**Context:** Before diving into Exercise 14.2 on FFT optimization, this example demonstrates how to implement, optimize, and benchmark different FFT approaches, showing the dramatic performance gains possible through algorithmic improvements.

1. **Theoretical Background:**
   - The naive DFT has O(N²) complexity, making it impractical for large datasets
   - FFT algorithms achieve O(N log N) through divide-and-conquer strategies
   - Different implementations can have vastly different performance characteristics
   - Understanding these differences is crucial for real-time applications

2. **Example:**
   Let's implement a comprehensive FFT optimization and benchmarking system.
   
   **Step 1:** Implement multiple FFT approaches with timing
   ```python
   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from scipy import fftpack
   import pandas as pd
   from numba import jit
   
   class FFTBenchmark:
       def __init__(self):
           self.results = {}
           
       def naive_dft(self, x):
           """O(N²) naive implementation for comparison"""
           N = len(x)
           X = np.zeros(N, dtype=complex)
           
           for k in range(N):
               for n in range(N):
                   X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
           
           return X
       
       @staticmethod
       @jit(nopython=True)
       def naive_dft_numba(x):
           """Numba-accelerated naive DFT"""
           N = len(x)
           X = np.zeros(N, dtype=np.complex128)
           
           for k in range(N):
               for n in range(N):
                   angle = -2 * np.pi * k * n / N
                   X[k] += x[n] * (np.cos(angle) + 1j * np.sin(angle))
           
           return X
       
       def recursive_fft(self, x):
           """Recursive Cooley-Tukey FFT implementation"""
           N = len(x)
           
           # Base case
           if N <= 1:
               return x
           
           # Ensure N is a power of 2 (pad with zeros if necessary)
           if N & (N - 1) != 0:
               next_power = 1 << (N - 1).bit_length()
               x_padded = np.zeros(next_power, dtype=complex)
               x_padded[:N] = x
               return self.recursive_fft(x_padded)[:N]
           
           # Divide
           even = self.recursive_fft(x[0::2])
           odd = self.recursive_fft(x[1::2])
           
           # Combine
           T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
           
           return [even[k] + T[k] for k in range(N // 2)] + \
                  [even[k] - T[k] for k in range(N // 2)]
       
       def iterative_fft(self, x):
           """Iterative in-place FFT implementation"""
           x = np.array(x, dtype=complex)
           N = len(x)
           
           # Bit-reverse copy
           j = 0
           for i in range(1, N):
               bit = N >> 1
               while j & bit:
                   j ^= bit
                   bit >>= 1
               j ^= bit
               if i < j:
                   x[i], x[j] = x[j], x[i]
           
           # Main FFT computation
           length = 2
           while length <= N:
               angle = -2 * np.pi / length
               wlen = complex(np.cos(angle), np.sin(angle))
               
               for i in range(0, N, length):
                   w = complex(1)
                   for j in range(length // 2):
                       u = x[i + j]
                       v = x[i + j + length // 2] * w
                       x[i + j] = u + v
                       x[i + j + length // 2] = u - v
                       w *= wlen
               
               length <<= 1
           
           return x
       
       def benchmark_methods(self, signal_lengths, n_trials=5):
           """Comprehensive benchmarking of FFT methods"""
           
           methods = {
               'Naive DFT': self.naive_dft,
               'Naive DFT (Numba)': self.naive_dft_numba,
               'Recursive FFT': self.recursive_fft,
               'Iterative FFT': self.iterative_fft,
               'NumPy FFT': np.fft.fft,
               'SciPy FFT': fftpack.fft
           }
           
           results = {
               'method': [],
               'signal_length': [],
               'time_seconds': [],
               'operations_per_second': [],
               'memory_efficiency': []
           }
           
           for length in signal_lengths:
               print(f"Testing signal length: {length}")
               
               # Generate test signal
               np.random.seed(42)
               test_signal = np.random.random(length) + 1j * np.random.random(length)
               
               for method_name, method_func in methods.items():
                   # Skip naive methods for large inputs
                   if length > 1024 and 'Naive' in method_name:
                       continue
                   
                   times = []
                   
                   for trial in range(n_trials):
                       start_time = time.perf_counter()
                       
                       try:
                           result = method_func(test_signal.copy())
                           end_time = time.perf_counter()
                           
                           # Verify correctness (compare with NumPy)
                           if method_name != 'NumPy FFT':
                               reference = np.fft.fft(test_signal)
                               if not np.allclose(result, reference, rtol=1e-10):
                                   print(f"Warning: {method_name} accuracy issue at length {length}")
                           
                           times.append(end_time - start_time)
                           
                       except Exception as e:
                           print(f"Error with {method_name} at length {length}: {e}")
                           times.append(float('inf'))
                           break
                   
                   if times and not np.isinf(times[0]):
                       avg_time = np.mean(times)
                       std_time = np.std(times)
                       
                       # Calculate theoretical operations
                       if 'Naive' in method_name:
                           ops = length ** 2  # O(N²)
                       else:
                           ops = length * np.log2(length)  # O(N log N)
                       
                       ops_per_sec = ops / avg_time if avg_time > 0 else 0
                       
                       results['method'].append(method_name)
                       results['signal_length'].append(length)
                       results['time_seconds'].append(avg_time)
                       results['operations_per_second'].append(ops_per_sec)
                       results['memory_efficiency'].append(1.0)  # Simplified
                       
                       print(f"  {method_name:20s}: {avg_time:.6f}s (±{std_time:.6f}s)")
           
           return pd.DataFrame(results)
       
       def analyze_complexity(self, results_df):
           """Analyze algorithmic complexity from benchmark results"""
           
           complexity_analysis = {}
           
           for method in results_df['method'].unique():
               method_data = results_df[results_df['method'] == method]
               
               if len(method_data) < 3:  # Need enough points for analysis
                   continue
               
               lengths = method_data['signal_length'].values
               times = method_data['time_seconds'].values
               
               # Fit different complexity models
               log_lengths = np.log2(lengths)
               log_times = np.log(times)
               
               # Linear model: log(T) = a + b*log(N)
               coeffs = np.polyfit(log_lengths, log_times, 1)
               complexity_exponent = coeffs[0]
               
               # R² for goodness of fit
               predicted_log_times = np.polyval(coeffs, log_lengths)
               r_squared = 1 - np.sum((log_times - predicted_log_times)**2) / np.sum((log_times - np.mean(log_times))**2)
               
               complexity_analysis[method] = {
                   'exponent': complexity_exponent,
                   'r_squared': r_squared,
                   'theoretical_complexity': 'O(N²)' if 'Naive' in method else 'O(N log N)'
               }
           
           return complexity_analysis
       
       def memory_profiling(self, signal_length=2**16):
           """Profile memory usage of different FFT implementations"""
           import tracemalloc
           
           np.random.seed(42)
           test_signal = np.random.random(signal_length) + 1j * np.random.random(signal_length)
           
           memory_results = {}
           
           methods = {
               'Iterative FFT (In-place)': self.iterative_fft,
               'Recursive FFT': self.recursive_fft,
               'NumPy FFT': np.fft.fft
           }
           
           for method_name, method_func in methods.items():
               tracemalloc.start()
               
               try:
                   result = method_func(test_signal.copy())
                   current, peak = tracemalloc.get_traced_memory()
                   
                   memory_results[method_name] = {
                       'current_mb': current / 1024 / 1024,
                       'peak_mb': peak / 1024 / 1024,
                       'efficiency_ratio': signal_length * 16 / peak  # bytes per complex number
                   }
                   
               except Exception as e:
                   print(f"Memory profiling error for {method_name}: {e}")
               
               finally:
                   tracemalloc.stop()
           
           return memory_results
   ```
   
   **Step 2:** Comprehensive benchmarking and analysis
   ```python
   # Initialize benchmark system
   benchmark = FFTBenchmark()
   
   # Define test sizes (powers of 2 for optimal FFT performance)
   test_lengths = [2**i for i in range(6, 16)]  # 64 to 32768
   
   # Run comprehensive benchmark
   print("Running FFT Benchmarks...")
   results_df = benchmark.benchmark_methods(test_lengths, n_trials=3)
   
   # Analyze algorithmic complexity
   complexity_analysis = benchmark.analyze_complexity(results_df)
   
   # Memory profiling
   memory_results = benchmark.memory_profiling(signal_length=2**14)
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(2, 3, figsize=(18, 12))
   
   # Performance comparison plot
   ax = axes[0, 0]
   
   for method in results_df['method'].unique():
       method_data = results_df[results_df['method'] == method]
       
       if len(method_data) > 1:
           ax.loglog(method_data['signal_length'], method_data['time_seconds'], 
                    'o-', label=method, alpha=0.8, markersize=4)
   
   # Add theoretical complexity lines
   lengths_smooth = np.logspace(np.log10(64), np.log10(32768), 100)
   
   # O(N²) reference
   n_squared_times = (lengths_smooth / 1000) ** 2 / 1e6  # Normalized
   ax.loglog(lengths_smooth, n_squared_times, '--', color='red', alpha=0.5, label='O(N²) reference')
   
   # O(N log N) reference  
   n_log_n_times = (lengths_smooth * np.log2(lengths_smooth)) / 1e6  # Normalized
   ax.loglog(lengths_smooth, n_log_n_times, '--', color='green', alpha=0.5, label='O(N log N) reference')
   
   ax.set_xlabel('Signal Length')
   ax.set_ylabel('Time (seconds)')
   ax.set_title('FFT Performance Comparison')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Operations per second comparison
   ax = axes[0, 1]
   
   # Focus on efficient methods only
   efficient_methods = [m for m in results_df['method'].unique() if 'Naive' not in m]
   
   method_ops = []
   method_names = []
   
   for method in efficient_methods:
       method_data = results_df[results_df['method'] == method]
       if len(method_data) > 0:
           avg_ops = method_data['operations_per_second'].mean()
           method_ops.append(avg_ops)
           method_names.append(method)
   
   bars = ax.bar(method_names, method_ops, alpha=0.8)
   ax.set_ylabel('Operations per Second')
   ax.set_title('Average Computational Throughput')
   ax.tick_params(axis='x', rotation=45)
   
   # Highlight best performers
   if method_ops:
       max_ops = max(method_ops)
       for bar, ops in zip(bars, method_ops):
           if ops > max_ops * 0.8:  # Within 20% of best
               bar.set_color('green')
   
   # Complexity analysis visualization
   ax = axes[0, 2]
   
   methods_with_analysis = list(complexity_analysis.keys())
   exponents = [complexity_analysis[m]['exponent'] for m in methods_with_analysis]
   r_squared_values = [complexity_analysis[m]['r_squared'] for m in methods_with_analysis]
   
   colors = ['red' if 'Naive' in m else 'blue' for m in methods_with_analysis]
   
   scatter = ax.scatter(exponents, r_squared_values, c=colors, alpha=0.8, s=100)
   
   for i, method in enumerate(methods_with_analysis):
       ax.annotate(method, (exponents[i], r_squared_values[i]), 
                  xytext=(5, 5), textcoords='offset points', fontsize=8)
   
   ax.axvline(1, color='green', linestyle='--', alpha=0.7, label='Linear (N)')
   ax.axvline(2, color='red', linestyle='--', alpha=0.7, label='Quadratic (N²)')
   ax.axhline(0.95, color='gray', linestyle=':', alpha=0.7, label='Good fit threshold')
   
   ax.set_xlabel('Fitted Complexity Exponent')
   ax.set_ylabel('R² (Goodness of Fit)')
   ax.set_title('Complexity Analysis')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Memory usage comparison
   ax = axes[1, 0]
   
   if memory_results:
       methods_mem = list(memory_results.keys())
       peak_memory = [memory_results[m]['peak_mb'] for m in methods_mem]
       current_memory = [memory_results[m]['current_mb'] for m in methods_mem]
       
       x = np.arange(len(methods_mem))
       width = 0.35
       
       ax.bar(x - width/2, current_memory, width, label='Current Memory', alpha=0.8)
       ax.bar(x + width/2, peak_memory, width, label='Peak Memory', alpha=0.8)
       
       ax.set_xlabel('Method')
       ax.set_ylabel('Memory Usage (MB)')
       ax.set_title('Memory Usage Comparison')
       ax.set_xticks(x)
       ax.set_xticklabels(methods_mem, rotation=45)
       ax.legend()
   
   # Scaling analysis
   ax = axes[1, 1]
   
   # Show how performance scales with problem size
   for method in ['NumPy FFT', 'Iterative FFT', 'Recursive FFT']:
       method_data = results_df[results_df['method'] == method]
       
       if len(method_data) > 3:
           lengths = method_data['signal_length'].values
           times = method_data['time_seconds'].values
           
           # Calculate speedup relative to smallest size
           speedup_ratio = times[0] * lengths / (times * lengths[0])
           
           ax.semilogx(lengths, speedup_ratio, 'o-', label=method, alpha=0.8)
   
   # Theoretical ideal scaling
   lengths_ref = np.array(test_lengths)
   ideal_scaling = lengths_ref / lengths_ref[0]  # Linear speedup
   ax.semilogx(lengths_ref, ideal_scaling, '--', color='red', alpha=0.5, 
              label='Ideal Linear Scaling')
   
   ax.set_xlabel('Signal Length')
   ax.set_ylabel('Relative Throughput')
   ax.set_title('Scaling Efficiency')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Error analysis
   ax = axes[1, 2]
   
   # Compare numerical accuracy of different methods
   np.random.seed(42)
   test_signal = np.random.random(1024) + 1j * np.random.random(1024)
   reference_result = np.fft.fft(test_signal)
   
   methods_to_test = {
       'Recursive FFT': benchmark.recursive_fft,
       'Iterative FFT': benchmark.iterative_fft,
       'SciPy FFT': fftpack.fft
   }
   
   errors = []
   method_names_error = []
   
   for method_name, method_func in methods_to_test.items():
       try:
           result = method_func(test_signal)
           
           # Calculate different error metrics
           abs_error = np.abs(result - reference_result)
           rel_error = abs_error / (np.abs(reference_result) + 1e-15)
           
           max_abs_error = np.max(abs_error)
           mean_rel_error = np.mean(rel_error)
           
           errors.append(mean_rel_error)
           method_names_error.append(method_name)
           
       except Exception as e:
           print(f"Error testing {method_name}: {e}")
   
   if errors:
       ax.semilogy(method_names_error, errors, 'bo-', markersize=8)
       ax.axhline(1e-10, color='red', linestyle='--', alpha=0.7, 
                 label='Machine precision threshold')
       
       ax.set_ylabel('Mean Relative Error')
       ax.set_title('Numerical Accuracy Comparison')
       ax.tick_params(axis='x', rotation=45)
       ax.legend()
       ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()
   
   # Print comprehensive analysis
   print("\nFFT Optimization Analysis:")
   print("=" * 50)
   
   print(f"\nPerformance Summary (for N=16384):")
   large_n_data = results_df[results_df['signal_length'] == 16384]
   
   if not large_n_data.empty:
       for method in large_n_data['method'].unique():
           method_data = large_n_data[large_n_data['method'] == method]
           time_val = method_data['time_seconds'].iloc[0]
           ops_val = method_data['operations_per_second'].iloc[0]
           
           print(f"  {method:20s}: {time_val:.6f}s ({ops_val:.2e} ops/s)")
   
   print(f"\nComplexity Analysis:")
   for method, analysis in complexity_analysis.items():
       theoretical = analysis['theoretical_complexity']
       measured_exp = analysis['exponent']
       fit_quality = analysis['r_squared']
       
       print(f"  {method:20s}: {theoretical} (measured: N^{measured_exp:.2f}, R²={fit_quality:.3f})")
   
   print(f"\nMemory Efficiency:")
   if memory_results:
       for method, mem_data in memory_results.items():
           peak_mb = mem_data['peak_mb']
           efficiency = mem_data['efficiency_ratio']
           print(f"  {method:20s}: {peak_mb:.2f} MB peak ({efficiency:.1f}x efficiency)")
   
   # Speed improvement analysis
   if len(results_df[results_df['method'] == 'NumPy FFT']) > 0:
       numpy_times = results_df[results_df['method'] == 'NumPy FFT']['time_seconds']
       
       print(f"\nSpeedup vs NumPy FFT:")
       for method in results_df['method'].unique():
           if method != 'NumPy FFT':
               method_times = results_df[results_df['method'] == method]['time_seconds']
               
               if len(method_times) > 0 and len(numpy_times) > 0:
                   # Compare on same-sized problems
                   common_lengths = set(results_df[results_df['method'] == method]['signal_length']) & \
                                  set(results_df[results_df['method'] == 'NumPy FFT']['signal_length'])
                   
                   if common_lengths:
                       speedups = []
                       for length in common_lengths:
                           numpy_time = results_df[(results_df['method'] == 'NumPy FFT') & 
                                                 (results_df['signal_length'] == length)]['time_seconds'].iloc[0]
                           method_time = results_df[(results_df['method'] == method) & 
                                                  (results_df['signal_length'] == length)]['time_seconds'].iloc[0]
                           speedups.append(numpy_time / method_time)
                       
                       avg_speedup = np.mean(speedups)
                       print(f"  {method:20s}: {avg_speedup:.2f}x")
   
   print(f"\nKey Insights:")
   print("- FFT algorithms provide dramatic O(N²) → O(N log N) improvement")
   print("- Library implementations (NumPy, SciPy) are highly optimized")
   print("- Memory-efficient in-place algorithms reduce memory overhead")
   print("- Numerical accuracy is generally excellent across implementations")
   
   # Practical recommendations
   fastest_method = results_df.loc[results_df['operations_per_second'].idxmax(), 'method']
   print(f"\nRecommendations:")
   print(f"- For production use: {fastest_method}")
   print("- For educational purposes: Recursive FFT (clear algorithm)")
   print("- For memory-constrained systems: Iterative FFT (in-place)")
   print("- Always use library implementations for real applications")
   ```
   
   **Step 3:** Real-world application example
   ```python
   # Demonstrate FFT optimization in practical signal processing
   def practical_fft_example():
       """Show FFT optimization in real signal processing context"""
       
       # Generate realistic signal
       fs = 1000  # Sampling frequency
       duration = 10  # seconds
       t = np.linspace(0, duration, fs * duration, endpoint=False)
       
       # Complex signal with multiple components
       signal = (2 * np.sin(2 * np.pi * 50 * t) +     # 50 Hz component
                1.5 * np.sin(2 * np.pi * 120 * t) +   # 120 Hz component  
                0.8 * np.sin(2 * np.pi * 200 * t) +   # 200 Hz component
                0.3 * np.random.randn(len(t)))         # Noise
       
       print(f"Processing signal: {len(signal):,} samples")
       
       # Time different FFT approaches
       methods = {
           'NumPy FFT': lambda x: np.fft.fft(x),
           'SciPy FFT': lambda x: fftpack.fft(x),
           'Our Iterative FFT': benchmark.iterative_fft
       }
       
       results = {}
       
       for name, method in methods.items():
           start_time = time.perf_counter()
           spectrum = method(signal)
           end_time = time.perf_counter()
           
           processing_time = end_time - start_time
           results[name] = {
               'time': processing_time,
               'spectrum': spectrum,
               'throughput': len(signal) / processing_time
           }
           
           print(f"{name:20s}: {processing_time:.6f}s ({len(signal)/processing_time:.0f} samples/s)")
       
       # Verify results are consistent
       reference = results['NumPy FFT']['spectrum']
       for name, data in results.items():
           if name != 'NumPy FFT':
               max_error = np.max(np.abs(data['spectrum'] - reference))
               print(f"{name} max error: {max_error:.2e}")
       
       # Visualize spectrum
       freqs = np.fft.fftfreq(len(signal), 1/fs)
       
       plt.figure(figsize=(12, 8))
       
       # Time domain
       plt.subplot(2, 2, 1)
       plt.plot(t[:1000], signal[:1000])  # Show first second
       plt.xlabel('Time (s)')
       plt.ylabel('Amplitude')
       plt.title('Input Signal (First Second)')
       
       # Frequency domain
       plt.subplot(2, 2, 2)
       spectrum_mag = np.abs(reference)
       plt.plot(freqs[:len(freqs)//2], spectrum_mag[:len(spectrum_mag)//2])
       plt.xlabel('Frequency (Hz)')
       plt.ylabel('Magnitude')
       plt.title('FFT Spectrum')
       plt.xlim(0, 300)
       
       # Performance comparison
       plt.subplot(2, 2, 3)
       names = list(results.keys())
       times = [results[name]['time'] for name in names]
       throughputs = [results[name]['throughput'] for name in names]
       
       bars = plt.bar(names, times)
       plt.ylabel('Processing Time (s)')
       plt.title('Performance Comparison')
       plt.xticks(rotation=45)
       
       # Throughput
       plt.subplot(2, 2, 4)
       plt.bar(names, throughputs)
       plt.ylabel('Samples/Second')
       plt.title('Throughput Comparison')
       plt.xticks(rotation=45)
       
       plt.tight_layout()
       plt.show()
       
       return results
   
   # Run practical example
   practical_results = practical_fft_example()
   ```

3. **Connection to Exercise 14.2:**
   This comprehensive FFT analysis provides the foundation for Exercise 14.2, demonstrating how to implement, optimize, and benchmark different FFT approaches while understanding the practical implications of algorithmic choices.

### Worked Example 2: Parallel Processing for Time Series Analysis
**Context:** This example demonstrates how to effectively parallelize time series computations, preparing students for Exercise 14.3 on distributed computing approaches.

1. **Theoretical Background:**
   - Time series analysis often involves embarrassingly parallel computations
   - Different parallelization strategies suit different problem types
   - Proper load balancing and overhead management are crucial for efficiency

2. **Example:**
   Let's implement comprehensive parallel processing strategies for time series.
   
   **Step 1:** Parallel processing framework
   ```python
   import numpy as np
   import pandas as pd
   import multiprocessing as mp
   from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
   import time
   from functools import partial
   import matplotlib.pyplot as plt
   
   class ParallelTimeSeriesProcessor:
       def __init__(self):
           self.n_cores = mp.cpu_count()
           
       def parallel_cross_correlation(self, time_series_list, method='process', chunk_size=None):
           """Compute cross-correlations in parallel"""
           
           n_series = len(time_series_list)
           
           # Generate all pairs
           pairs = [(i, j) for i in range(n_series) for j in range(i+1, n_series)]
           
           def compute_single_correlation(pair):
               i, j = pair
               return i, j, np.corrcoef(time_series_list[i], time_series_list[j])[0, 1]
           
           if method == 'sequential':
               results = [compute_single_correlation(pair) for pair in pairs]
               
           elif method == 'process':
               with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                   results = list(executor.map(compute_single_correlation, pairs))
                   
           elif method == 'thread':
               with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                   results = list(executor.map(compute_single_correlation, pairs))
           
           # Construct correlation matrix
           correlation_matrix = np.eye(n_series)
           for i, j, corr in results:
               correlation_matrix[i, j] = corr
               correlation_matrix[j, i] = corr
               
           return correlation_matrix
       
       def parallel_rolling_statistics(self, time_series, window_sizes, statistics=['mean', 'std', 'skew'], 
                                     method='process'):
           """Compute rolling statistics in parallel"""
           
           def compute_rolling_stat(args):
               ts, window, stat = args
               
               if stat == 'mean':
                   return pd.Series(ts).rolling(window).mean().values
               elif stat == 'std':
                   return pd.Series(ts).rolling(window).std().values
               elif stat == 'skew':
                   return pd.Series(ts).rolling(window).skew().values
               elif stat == 'kurt':
                   return pd.Series(ts).rolling(window).kurt().values
           
           # Create tasks
           tasks = [(time_series, window, stat) 
                   for window in window_sizes 
                   for stat in statistics]
           
           if method == 'sequential':
               results = [compute_rolling_stat(task) for task in tasks]
           elif method == 'process':
               with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                   results = list(executor.map(compute_rolling_stat, tasks))
           elif method == 'thread':
               with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                   results = list(executor.map(compute_rolling_stat, tasks))
           
           # Organize results
           organized_results = {}
           idx = 0
           for window in window_sizes:
               organized_results[window] = {}
               for stat in statistics:
                   organized_results[window][stat] = results[idx]
                   idx += 1
                   
           return organized_results
       
       def parallel_monte_carlo_simulation(self, model_func, n_simulations=10000, 
                                         simulation_length=1000, method='process'):
           """Parallel Monte Carlo simulation for time series models"""
           
           def run_simulation(seed):
               np.random.seed(seed)
               return model_func(simulation_length)
           
           seeds = np.random.randint(0, 2**31, n_simulations)
           
           if method == 'sequential':
               simulations = [run_simulation(seed) for seed in seeds]
           elif method == 'process':
               with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                   simulations = list(executor.map(run_simulation, seeds))
           elif method == 'thread':
               with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                   simulations = list(executor.map(run_simulation, seeds))
           
           return np.array(simulations)
       
       def parallel_spectral_analysis(self, time_series_list, method='process'):
           """Parallel spectral analysis of multiple time series"""
           
           def compute_spectrum(ts):
               # Compute power spectral density
               freqs = np.fft.fftfreq(len(ts))
               fft_vals = np.fft.fft(ts)
               psd = np.abs(fft_vals)**2
               
               # Find dominant frequency
               positive_freqs = freqs[:len(freqs)//2]
               positive_psd = psd[:len(psd)//2]
               dominant_freq_idx = np.argmax(positive_psd[1:]) + 1  # Skip DC component
               dominant_freq = positive_freqs[dominant_freq_idx]
               
               return {
                   'frequencies': freqs,
                   'psd': psd,
                   'dominant_frequency': dominant_freq,
                   'total_power': np.sum(psd),
                   'spectral_centroid': np.sum(positive_freqs * positive_psd) / np.sum(positive_psd)
               }
           
           if method == 'sequential':
               results = [compute_spectrum(ts) for ts in time_series_list]
           elif method == 'process':
               with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                   results = list(executor.map(compute_spectrum, time_series_list))
           elif method == 'thread':
               with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                   results = list(executor.map(compute_spectrum, time_series_list))
           
           return results
       
       def benchmark_parallel_methods(self, test_sizes=[10, 50, 100], n_trials=3):
           """Benchmark different parallelization approaches"""
           
           results = {
               'task_type': [],
               'data_size': [],
               'method': [],
               'time_seconds': [],
               'speedup': [],
               'efficiency': []
           }
           
           for size in test_sizes:
               print(f"Benchmarking with size: {size}")
               
               # Generate test data
               np.random.seed(42)
               test_series_list = [np.random.randn(1000) for _ in range(size)]
               
               # Test cross-correlation
               for method in ['sequential', 'process', 'thread']:
                   times = []
                   
                   for trial in range(n_trials):
                       start_time = time.perf_counter()
                       _ = self.parallel_cross_correlation(test_series_list, method=method)
                       end_time = time.perf_counter()
                       times.append(end_time - start_time)
                   
                   avg_time = np.mean(times)
                   
                   results['task_type'].append('Cross-correlation')
                   results['data_size'].append(size)
                   results['method'].append(method)
                   results['time_seconds'].append(avg_time)
                   
                   print(f"  Cross-correlation {method:12s}: {avg_time:.4f}s")
               
               # Calculate speedups
               seq_time = None
               for i, method in enumerate(['sequential', 'process', 'thread']):
                   idx = -(3-i)  # Last 3 entries
                   time_val = results['time_seconds'][idx]
                   
                   if method == 'sequential':
                       seq_time = time_val
                       speedup = 1.0
                   else:
                       speedup = seq_time / time_val if time_val > 0 else 0
                   
                   results['speedup'].append(speedup)
                   results['efficiency'].append(speedup / self.n_cores)
               
               # Test rolling statistics
               window_sizes = [10, 20, 50]
               single_series = test_series_list[0]
               
               for method in ['sequential', 'process', 'thread']:
                   times = []
                   
                   for trial in range(n_trials):
                       start_time = time.perf_counter()
                       _ = self.parallel_rolling_statistics(single_series, window_sizes, method=method)
                       end_time = time.perf_counter()
                       times.append(end_time - start_time)
                   
                   avg_time = np.mean(times)
                   
                   results['task_type'].append('Rolling Statistics')
                   results['data_size'].append(size)
                   results['method'].append(method)
                   results['time_seconds'].append(avg_time)
                   
                   print(f"  Rolling stats {method:12s}: {avg_time:.4f}s")
               
               # Calculate speedups for rolling statistics
               seq_time = None
               for i, method in enumerate(['sequential', 'process', 'thread']):
                   idx = -(3-i)
                   time_val = results['time_seconds'][idx]
                   
                   if method == 'sequential':
                       seq_time = time_val
                       speedup = 1.0
                   else:
                       speedup = seq_time / time_val if time_val > 0 else 0
                   
                   results['speedup'].append(speedup)
                   results['efficiency'].append(speedup / self.n_cores)
           
           return pd.DataFrame(results)
   ```
   
   **Step 2:** Apply parallel processing analysis
   ```python
   # Initialize parallel processor
   processor = ParallelTimeSeriesProcessor()
   print(f"Available CPU cores: {processor.n_cores}")
   
   # Benchmark parallel methods
   benchmark_results = processor.benchmark_parallel_methods(test_sizes=[20, 50, 100], n_trials=3)
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(2, 3, figsize=(18, 12))
   
   # Speedup comparison
   ax = axes[0, 0]
   
   for task_type in benchmark_results['task_type'].unique():
       task_data = benchmark_results[benchmark_results['task_type'] == task_type]
       
       for method in ['process', 'thread']:
           method_data = task_data[task_data['method'] == method]
           if len(method_data) > 0:
               ax.plot(method_data['data_size'], method_data['speedup'], 
                      'o-', label=f'{task_type} - {method}', alpha=0.8)
   
   ax.axhline(y=processor.n_cores, color='red', linestyle='--', alpha=0.7, 
             label=f'Theoretical Maximum ({processor.n_cores}x)')
   ax.set_xlabel('Data Size')
   ax.set_ylabel('Speedup Factor')
   ax.set_title('Parallel Speedup Analysis')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Efficiency comparison
   ax = axes[0, 1]
   
   for task_type in benchmark_results['task_type'].unique():
       task_data = benchmark_results[benchmark_results['task_type'] == task_type]
       
       for method in ['process', 'thread']:
           method_data = task_data[task_data['method'] == method]
           if len(method_data) > 0:
               ax.plot(method_data['data_size'], method_data['efficiency'], 
                      'o-', label=f'{task_type} - {method}', alpha=0.8)
   
   ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
   ax.set_xlabel('Data Size')
   ax.set_ylabel('Parallel Efficiency')
   ax.set_title('Parallel Efficiency Analysis')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Execution time comparison
   ax = axes[0, 2]
   
   # Focus on largest data size for comparison
   large_data = benchmark_results[benchmark_results['data_size'] == benchmark_results['data_size'].max()]
   
   grouped_data = large_data.groupby(['task_type', 'method'])['time_seconds'].mean().unstack()
   grouped_data.plot(kind='bar', ax=ax, alpha=0.8)
   
   ax.set_ylabel('Execution Time (seconds)')
   ax.set_title(f'Execution Time Comparison\n(Data Size: {benchmark_results["data_size"].max()})')
   ax.tick_params(axis='x', rotation=45)
   ax.legend(title='Method')
   
   # Demonstrate parallel Monte Carlo
   print("\nRunning Parallel Monte Carlo Simulation...")
   
   def ar1_model(length, phi=0.8, sigma=1.0):
       """Generate AR(1) time series"""
       y = np.zeros(length)
       y[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))
       
       for t in range(1, length):
           y[t] = phi * y[t-1] + np.random.normal(0, sigma)
       
       return y
   
   # Time different Monte Carlo approaches
   n_sims = 1000
   sim_length = 1000
   
   mc_results = {}
   
   for method in ['sequential', 'process', 'thread']:
       start_time = time.perf_counter()
       simulations = processor.parallel_monte_carlo_simulation(
           ar1_model, n_simulations=n_sims, simulation_length=sim_length, method=method
       )
       end_time = time.perf_counter()
       
       mc_results[method] = {
           'time': end_time - start_time,
           'simulations': simulations
       }
       
       print(f"Monte Carlo {method:12s}: {end_time - start_time:.4f}s")
   
   # Monte Carlo results analysis
   ax = axes[1, 0]
   
   # Plot some sample paths
   for i in range(5):
       ax.plot(mc_results['process']['simulations'][i], alpha=0.6)
   
   ax.set_xlabel('Time')
   ax.set_ylabel('Value')
   ax.set_title('Sample AR(1) Paths from Monte Carlo')
   
   # Distribution analysis
   ax = axes[1, 1]
   
   # Final values distribution
   final_values = mc_results['process']['simulations'][:, -1]
   
   ax.hist(final_values, bins=50, alpha=0.7, density=True)
   ax.axvline(np.mean(final_values), color='red', linestyle='--', 
             label=f'Mean: {np.mean(final_values):.2f}')
   ax.axvline(np.median(final_values), color='green', linestyle='--', 
             label=f'Median: {np.median(final_values):.2f}')
   
   ax.set_xlabel('Final Value')
   ax.set_ylabel('Density')
   ax.set_title('Distribution of Final Values')
   ax.legend()
   
   # Performance summary
   ax = axes[1, 2]
   
   methods = list(mc_results.keys())
   times = [mc_results[method]['time'] for method in methods]
   speedups = [times[0] / time for time in times]  # Relative to sequential
   
   x = np.arange(len(methods))
   
   ax2 = ax.twinx()
   
   bars = ax.bar(x, times, alpha=0.8, label='Execution Time')
   line = ax2.plot(x, speedups, 'ro-', markersize=8, label='Speedup')
   
   ax.set_xlabel('Method')
   ax.set_ylabel('Time (seconds)', color='blue')
   ax2.set_ylabel('Speedup Factor', color='red')
   ax.set_title('Monte Carlo Performance')
   ax.set_xticks(x)
   ax.set_xticklabels(methods)
   
   # Combine legends
   lines1, labels1 = ax.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
   
   plt.tight_layout()
   plt.show()
   
   # Demonstrate advanced parallel spectral analysis
   print("\nRunning Parallel Spectral Analysis...")
   
   # Generate test signals with different characteristics
   np.random.seed(42)
   n_signals = 50
   signal_length = 2000
   
   test_signals = []
   
   for i in range(n_signals):
       t = np.linspace(0, 10, signal_length)
       
       # Different frequency components for each signal
       freq1 = 10 + i * 2  # Varying base frequency
       freq2 = 50 + np.random.uniform(-10, 10)  # Random second component
       
       signal = (np.sin(2 * np.pi * freq1 * t) + 
                0.5 * np.sin(2 * np.pi * freq2 * t) +
                0.1 * np.random.randn(signal_length))
       
       test_signals.append(signal)
   
   # Time spectral analysis
   spectral_results = {}
   
   for method in ['sequential', 'process', 'thread']:
       start_time = time.perf_counter()
       spectra = processor.parallel_spectral_analysis(test_signals, method=method)
       end_time = time.perf_counter()
       
       spectral_results[method] = {
           'time': end_time - start_time,
           'spectra': spectra
       }
       
       print(f"Spectral analysis {method:12s}: {end_time - start_time:.4f}s")
   
   # Analyze spectral results
   spectra = spectral_results['process']['spectra']
   dominant_freqs = [spectrum['dominant_frequency'] for spectrum in spectra]
   total_powers = [spectrum['total_power'] for spectrum in spectra]
   spectral_centroids = [spectrum['spectral_centroid'] for spectrum in spectra]
   
   # Additional visualization
   plt.figure(figsize=(15, 10))
   
   # Dominant frequency distribution
   plt.subplot(2, 3, 1)
   plt.scatter(range(n_signals), dominant_freqs, alpha=0.7)
   plt.xlabel('Signal Index')
   plt.ylabel('Dominant Frequency')
   plt.title('Dominant Frequencies Across Signals')
   
   # Total power distribution
   plt.subplot(2, 3, 2)
   plt.hist(total_powers, bins=20, alpha=0.7)
   plt.xlabel('Total Power')
   plt.ylabel('Count')
   plt.title('Distribution of Total Power')
   
   # Spectral centroid vs total power
   plt.subplot(2, 3, 3)
   plt.scatter(total_powers, spectral_centroids, alpha=0.7)
   plt.xlabel('Total Power')
   plt.ylabel('Spectral Centroid')
   plt.title('Spectral Characteristics')
   
   # Sample spectrum
   plt.subplot(2, 3, 4)
   sample_spectrum = spectra[0]
   freqs = sample_spectrum['frequencies'][:signal_length//2]
   psd = sample_spectrum['psd'][:signal_length//2]
   
   plt.plot(freqs, psd)
   plt.xlabel('Frequency')
   plt.ylabel('Power Spectral Density')
   plt.title('Sample Power Spectrum')
   
   # Performance comparison for spectral analysis
   plt.subplot(2, 3, 5)
   methods = list(spectral_results.keys())
   times = [spectral_results[method]['time'] for method in methods]
   speedups = [times[0] / time for time in times]
   
   x = np.arange(len(methods))
   
   plt.bar(x, times, alpha=0.8, label='Execution Time')
   plt.xlabel('Method')
   plt.ylabel('Time (seconds)')
   plt.title('Spectral Analysis Performance')
   plt.xticks(x, methods)
   
   # Overall performance summary
   plt.subplot(2, 3, 6)
   
   all_tasks = ['Cross-correlation', 'Rolling Statistics', 'Monte Carlo', 'Spectral Analysis']
   process_speedups = []
   thread_speedups = []
   
   # Collect speedups from benchmark
   cc_data = benchmark_results[(benchmark_results['task_type'] == 'Cross-correlation') & 
                              (benchmark_results['data_size'] == 100)]
   rs_data = benchmark_results[(benchmark_results['task_type'] == 'Rolling Statistics') & 
                              (benchmark_results['data_size'] == 100)]
   
   process_speedups.extend([
       cc_data[cc_data['method'] == 'process']['speedup'].iloc[0] if len(cc_data[cc_data['method'] == 'process']) > 0 else 0,
       rs_data[rs_data['method'] == 'process']['speedup'].iloc[0] if len(rs_data[rs_data['method'] == 'process']) > 0 else 0,
       times[0] / spectral_results['process']['time'],
       times[0] / mc_results['process']['time']
   ])
   
   thread_speedups.extend([
       cc_data[cc_data['method'] == 'thread']['speedup'].iloc[0] if len(cc_data[cc_data['method'] == 'thread']) > 0 else 0,
       rs_data[rs_data['method'] == 'thread']['speedup'].iloc[0] if len(rs_data[rs_data['method'] == 'thread']) > 0 else 0,
       times[0] / spectral_results['thread']['time'],
       times[0] / mc_results['thread']['time']
   ])
   
   x = np.arange(len(all_tasks))
   width = 0.35
   
   plt.bar(x - width/2, process_speedups, width, label='Process-based', alpha=0.8)
   plt.bar(x + width/2, thread_speedups, width, label='Thread-based', alpha=0.8)
   
   plt.axhline(y=processor.n_cores, color='red', linestyle='--', alpha=0.7, 
              label=f'Theoretical Max ({processor.n_cores}x)')
   
   plt.xlabel('Task Type')
   plt.ylabel('Speedup Factor')
   plt.title('Overall Parallelization Performance')
   plt.xticks(x, all_tasks, rotation=45)
   plt.legend()
   
   plt.tight_layout()
   plt.show()
   
   # Print comprehensive analysis
   print(f"\nParallel Processing Analysis:")
   print("=" * 50)
   
   print(f"\nSystem Information:")
   print(f"  CPU cores available: {processor.n_cores}")
   print(f"  Theoretical maximum speedup: {processor.n_cores}x")
   
   print(f"\nTask-specific Performance:")
   
   # Cross-correlation analysis
   cc_large = benchmark_results[(benchmark_results['task_type'] == 'Cross-correlation') & 
                               (benchmark_results['data_size'] == 100)]
   if len(cc_large) > 0:
       seq_time = cc_large[cc_large['method'] == 'sequential']['time_seconds'].iloc[0]
       process_time = cc_large[cc_large['method'] == 'process']['time_seconds'].iloc[0]
       thread_time = cc_large[cc_large['method'] == 'thread']['time_seconds'].iloc[0]
       
       print(f"\n  Cross-correlation (100 series):")
       print(f"    Sequential: {seq_time:.4f}s")
       print(f"    Process:    {process_time:.4f}s ({seq_time/process_time:.1f}x speedup)")
       print(f"    Thread:     {thread_time:.4f}s ({seq_time/thread_time:.1f}x speedup)")
       print(f"    Best method: {'Process' if process_time < thread_time else 'Thread'}")
   
   # Monte Carlo analysis
   seq_mc_time = mc_results['sequential']['time']
   process_mc_time = mc_results['process']['time']
   thread_mc_time = mc_results['thread']['time']
   
   print(f"\n  Monte Carlo ({n_sims} simulations):")
   print(f"    Sequential: {seq_mc_time:.4f}s")
   print(f"    Process:    {process_mc_time:.4f}s ({seq_mc_time/process_mc_time:.1f}x speedup)")
   print(f"    Thread:     {thread_mc_time:.4f}s ({seq_mc_time/thread_mc_time:.1f}x speedup)")
   print(f"    Best method: {'Process' if process_mc_time < thread_mc_time else 'Thread'}")
   
   # Spectral analysis
   seq_spec_time = spectral_results['sequential']['time']
   process_spec_time = spectral_results['process']['time']
   thread_spec_time = spectral_results['thread']['time']
   
   print(f"\n  Spectral Analysis ({n_signals} signals):")
   print(f"    Sequential: {seq_spec_time:.4f}s")
   print(f"    Process:    {process_spec_time:.4f}s ({seq_spec_time/process_spec_time:.1f}x speedup)")
   print(f"    Thread:     {thread_spec_time:.4f}s ({seq_spec_time/thread_spec_time:.1f}x speedup)")
   print(f"    Best method: {'Process' if process_spec_time < thread_spec_time else 'Thread'}")
   
   print(f"\nKey Insights:")
   print("- Process-based parallelization generally outperforms thread-based for CPU-intensive tasks")
   print("- Speedup depends heavily on problem size and computational complexity")
   print("- Overhead costs can dominate for small problems")
   print("- Different task types benefit from different parallelization strategies")
   
   print(f"\nRecommendations:")
   print("- Use process-based parallelization for CPU-bound computations")
   print("- Use thread-based parallelization for I/O-bound tasks")
   print("- Consider task granularity when choosing parallelization strategy")
   print("- Always benchmark with realistic problem sizes")
   ```

3. **Connection to Exercise 14.3:**
   This comprehensive parallel processing analysis provides the foundation for Exercise 14.3, demonstrating how to implement and optimize parallel algorithms while understanding the trade-offs between different approaches.

### Worked Example 3: Streaming Algorithms for Real-time Analysis
**Context:** This example demonstrates efficient algorithms for processing streaming time series data, essential for real-time applications and Exercise 14.4.

1. **Theoretical Background:**
   - Streaming algorithms process data in a single pass with limited memory
   - Online algorithms update statistics incrementally without storing all historical data
   - Approximation algorithms trade accuracy for efficiency when exact computation is impractical

2. **Example:**
   Let's implement comprehensive streaming algorithms for time series analysis.
   
   **Step 1:** Streaming algorithm framework
   ```python
   import numpy as np
   from collections import deque
   import matplotlib.pyplot as plt
   import time
   import pandas as pd
   
   class StreamingProcessor:
       def __init__(self, window_size=1000):
           self.window_size = window_size
           self.reset()
           
       def reset(self):
           """Reset all streaming statistics"""
           self.count = 0
           self.sum = 0.0
           self.sum_sq = 0.0
           self.min_val = float('inf')
           self.max_val = float('-inf')
           
           # For windowed statistics
           self.window = deque(maxlen=self.window_size)
           
           # For online variance (Welford's algorithm)
           self.mean = 0.0
           self.m2 = 0.0
           
           # For streaming quantiles (P² algorithm)
           self.quantile_positions = [1, 2, 3, 4, 5]
           self.quantile_heights = []
           self.quantile_n = [1, 2, 3, 4, 5]
           self.quantile_np = [1, 1+2*0.5, 1+4*0.5, 1+4*0.75, 5]  # For median (p=0.5)
           self.quantile_dn = [0, 0.5/4, 0.5/2, (1+0.5)/2, 1/4]
           
           # For streaming mode detection
           self.value_counts = {}
           self.mode_value = None
           self.mode_count = 0
           
           # For change point detection
           self.cusum_pos = 0.0
           self.cusum_neg = 0.0
           self.baseline_mean = None
           self.change_threshold = 5.0
           
       def update(self, value):
           """Update all streaming statistics with new value"""
           self.count += 1
           self.sum += value
           self.sum_sq += value * value
           
           # Min/max
           self.min_val = min(self.min_val, value)
           self.max_val = max(self.max_val, value)
           
           # Add to sliding window
           self.window.append(value)
           
           # Online mean and variance (Welford's algorithm)
           delta = value - self.mean
           self.mean += delta / self.count
           delta2 = value - self.mean
           self.m2 += delta * delta2
           
           # Update streaming quantiles (simplified P² algorithm)
           self._update_quantiles(value)
           
           # Update mode tracking
           self._update_mode(value)
           
           # Update change point detection
           self._update_cusum(value)
           
       def _update_quantiles(self, value):
           """Update streaming quantiles using P² algorithm (simplified)"""
           if len(self.quantile_heights) < 5:
               self.quantile_heights.append(value)
               if len(self.quantile_heights) == 5:
                   self.quantile_heights.sort()
           else:
               # Find position to insert new value
               k = 0
               for i, height in enumerate(self.quantile_heights):
                   if value < height:
                       k = i
                       break
               else:
                   k = 5
               
               # Update positions
               for i in range(k, 5):
                   self.quantile_n[i] += 1
               
               # Update desired positions
               for i in range(5):
                   self.quantile_np[i] += self.quantile_dn[i]
               
               # Adjust heights (simplified)
               for i in range(1, 4):  # Only adjust inner quantiles
                   d = self.quantile_np[i] - self.quantile_n[i]
                   if (d >= 1 and self.quantile_n[i+1] - self.quantile_n[i] > 1) or \
                      (d <= -1 and self.quantile_n[i-1] - self.quantile_n[i] < -1):
                       
                       # Parabolic formula (simplified)
                       new_height = self.quantile_heights[i] + d / (self.quantile_n[i+1] - self.quantile_n[i-1]) * \
                                   ((self.quantile_n[i] - self.quantile_n[i-1] + d) * 
                                    (self.quantile_heights[i+1] - self.quantile_heights[i]) / 
                                    (self.quantile_n[i+1] - self.quantile_n[i]) + 
                                    (self.quantile_n[i+1] - self.quantile_n[i] - d) * 
                                    (self.quantile_heights[i] - self.quantile_heights[i-1]) / 
                                    (self.quantile_n[i] - self.quantile_n[i-1]))
                       
                       if self.quantile_heights[i-1] < new_height < self.quantile_heights[i+1]:
                           self.quantile_heights[i] = new_height
                           self.quantile_n[i] += 1 if d > 0 else -1
           
       def _update_mode(self, value):
           """Update streaming mode estimation"""
           # Discretize continuous values for mode calculation
           discretized = round(value, 1)  # Round to 1 decimal place
           
           if discretized not in self.value_counts:
               self.value_counts[discretized] = 0
           
           self.value_counts[discretized] += 1
           
           if self.value_counts[discretized] > self.mode_count:
               self.mode_value = discretized
               self.mode_count = self.value_counts[discretized]
               
       def _update_cusum(self, value):
           """Update CUSUM for change point detection"""
           if self.baseline_mean is None:
               if self.count > 50:  # Wait for initial baseline
                   self.baseline_mean = self.mean
           else:
               # CUSUM for detecting upward and downward changes
               deviation = value - self.baseline_mean
               self.cusum_pos = max(0, self.cusum_pos + deviation - 0.5)
               self.cusum_neg = max(0, self.cusum_neg - deviation - 0.5)
               
       def get_statistics(self):
           """Get current streaming statistics"""
           if self.count == 0:
               return {}
           
           variance = self.m2 / (self.count - 1) if self.count > 1 else 0
           
           stats = {
               'count': self.count,
               'mean': self.mean,
               'variance': variance,
               'std': np.sqrt(variance),
               'min': self.min_val,
               'max': self.max_val,
               'range': self.max_val - self.min_val,
               'sum': self.sum,
               'sum_sq': self.sum_sq
           }
           
           # Windowed statistics
           if len(self.window) > 0:
               window_array = np.array(self.window)
               stats['window_mean'] = np.mean(window_array)
               stats['window_std'] = np.std(window_array)
               stats['window_median'] = np.median(window_array)
               
           # Streaming quantiles
           if len(self.quantile_heights) == 5:
               stats['q25'] = self.quantile_heights[1]
               stats['q50'] = self.quantile_heights[2]
               stats['q75'] = self.quantile_heights[3]
               
           # Mode
           stats['mode'] = self.mode_value
           stats['mode_count'] = self.mode_count
           
           # Change point indicators
           stats['cusum_pos'] = self.cusum_pos
           stats['cusum_neg'] = self.cusum_neg
           stats['change_detected'] = (self.cusum_pos > self.change_threshold or 
                                     self.cusum_neg > self.change_threshold)
           
           return stats
   
   class StreamingAnomalyDetector:
       def __init__(self, window_size=100, threshold_factor=3.0):
           self.window_size = window_size
           self.threshold_factor = threshold_factor
           self.reset()
           
       def reset(self):
           self.processor = StreamingProcessor(self.window_size)
           self.anomaly_count = 0
           self.anomaly_history = deque(maxlen=self.window_size)
           
       def detect(self, value):
           """Detect if current value is anomalous"""
           self.processor.update(value)
           stats = self.processor.get_statistics()
           
           is_anomaly = False
           anomaly_score = 0.0
           
           if stats['count'] > 10:  # Need some data for reliable detection
               # Z-score based detection
               if stats['std'] > 0:
                   z_score = abs(value - stats['mean']) / stats['std']
                   if z_score > self.threshold_factor:
                       is_anomaly = True
                       anomaly_score = z_score
               
               # Additional checks using windowed statistics
               if len(self.processor.window) > 5:
                   if (value > stats['window_mean'] + self.threshold_factor * stats['window_std'] or
                       value < stats['window_mean'] - self.threshold_factor * stats['window_std']):
                       is_anomaly = True
                       anomaly_score = max(anomaly_score, abs(value - stats['window_mean']) / stats['window_std'])
           
           self.anomaly_history.append(is_anomaly)
           if is_anomaly:
               self.anomaly_count += 1
               
           return is_anomaly, anomaly_score
   
   class StreamingForecaster:
       def __init__(self, horizon=1, method='exponential_smoothing'):
           self.horizon = horizon
           self.method = method
           self.reset()
           
       def reset(self):
           self.count = 0
           self.last_value = 0.0
           self.trend = 0.0
           
           # Exponential smoothing parameters
           self.alpha = 0.3  # Level smoothing
           self.beta = 0.3   # Trend smoothing
           self.level = None
           
           # Simple moving average
           self.window = deque(maxlen=20)
           
       def update_and_forecast(self, value):
           """Update model and generate forecast"""
           self.count += 1
           
           if self.method == 'exponential_smoothing':
               return self._exponential_smoothing_forecast(value)
           elif self.method == 'moving_average':
               return self._moving_average_forecast(value)
           elif self.method == 'linear_trend':
               return self._linear_trend_forecast(value)
           
       def _exponential_smoothing_forecast(self, value):
           """Holt's exponential smoothing"""
           if self.level is None:
               self.level = value
               self.trend = 0.0
               return value  # First forecast is just the current value
           
           # Update level and trend
           prev_level = self.level
           self.level = self.alpha * value + (1 - self.alpha) * (prev_level + self.trend)
           self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
           
           # Forecast
           forecast = self.level + self.horizon * self.trend
           return forecast
           
       def _moving_average_forecast(self, value):
           """Simple moving average forecast"""
           self.window.append(value)
           
           if len(self.window) < 2:
               return value
               
           return np.mean(self.window)
           
       def _linear_trend_forecast(self, value):
           """Simple linear trend extrapolation"""
           if self.count == 1:
               self.last_value = value
               return value
           
           # Estimate trend
           self.trend = 0.1 * (value - self.last_value) + 0.9 * self.trend
           self.last_value = value
           
           return value + self.horizon * self.trend
   ```
   
   **Step 2:** Apply streaming algorithms to real-time data
   ```python
   # Generate streaming time series data with various characteristics
   def generate_streaming_data(n_samples=10000, change_points=None, anomaly_rate=0.02):
       """Generate realistic streaming data"""
       np.random.seed(42)
       
       data = []
       current_mean = 0.0
       current_std = 1.0
       trend = 0.0
       
       change_points = change_points or [2000, 5000, 8000]
       
       for i in range(n_samples):
           # Check for regime changes
           if i in change_points:
               if i == change_points[0]:
                   current_mean += 2.0  # Mean shift
               elif i == change_points[1]:
                   current_std *= 2.0   # Variance change
               elif i == change_points[2]:
                   trend += 0.001       # Trend change
           
           # Generate base value
           value = current_mean + trend * i + np.random.normal(0, current_std)
           
           # Add seasonal component
           seasonal = 0.5 * np.sin(2 * np.pi * i / 100) + 0.3 * np.sin(2 * np.pi * i / 200)
           value += seasonal
           
           # Inject random anomalies
           if np.random.random() < anomaly_rate:
               value += np.random.choice([-1, 1]) * np.random.uniform(5, 10)
           
           data.append(value)
           
       return data
   
   # Initialize streaming components
   processor = StreamingProcessor(window_size=200)
   anomaly_detector = StreamingAnomalyDetector(window_size=100, threshold_factor=2.5)
   forecaster = StreamingForecaster(horizon=1, method='exponential_smoothing')
   
   # Generate streaming data
   streaming_data = generate_streaming_data(n_samples=5000, 
                                          change_points=[1000, 2500, 4000])
   
   # Process streaming data and collect results
   results = {
       'timestamp': [],
       'value': [],
       'mean': [],
       'std': [],
       'window_mean': [],
       'window_std': [],
       'q25': [], 'q50': [], 'q75': [],
       'is_anomaly': [],
       'anomaly_score': [],
       'forecast': [],
       'cusum_pos': [],
       'cusum_neg': [],
       'change_detected': []
   }
   
   print("Processing streaming data...")
   
   # Simulate real-time processing
   start_time = time.time()
   
   for i, value in enumerate(streaming_data):
       # Update streaming processor
       processor.update(value)
       stats = processor.get_statistics()
       
       # Detect anomalies
       is_anomaly, anomaly_score = anomaly_detector.detect(value)
       
       # Generate forecast
       forecast = forecaster.update_and_forecast(value)
       
       # Store results
       results['timestamp'].append(i)
       results['value'].append(value)
       results['mean'].append(stats.get('mean', 0))
       results['std'].append(stats.get('std', 0))
       results['window_mean'].append(stats.get('window_mean', 0))
       results['window_std'].append(stats.get('window_std', 0))
       results['q25'].append(stats.get('q25', 0))
       results['q50'].append(stats.get('q50', 0))
       results['q75'].append(stats.get('q75', 0))
       results['is_anomaly'].append(is_anomaly)
       results['anomaly_score'].append(anomaly_score)
       results['forecast'].append(forecast)
       results['cusum_pos'].append(stats.get('cusum_pos', 0))
       results['cusum_neg'].append(stats.get('cusum_neg', 0))
       results['change_detected'].append(stats.get('change_detected', False))
       
       # Simulate processing time (optional)
       if i % 1000 == 0:
           print(f"Processed {i:,} samples...")
   
   processing_time = time.time() - start_time
   print(f"Total processing time: {processing_time:.3f}s")
   print(f"Processing rate: {len(streaming_data)/processing_time:.0f} samples/second")
   
   # Convert results to DataFrame for analysis
   results_df = pd.DataFrame(results)
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(3, 2, figsize=(16, 12))
   
   # Raw data with streaming statistics
   ax = axes[0, 0]
   
   ax.plot(results_df['timestamp'], results_df['value'], 'b-', alpha=0.6, linewidth=0.5, label='Data')
   ax.plot(results_df['timestamp'], results_df['mean'], 'r-', linewidth=2, label='Streaming Mean')
   ax.plot(results_df['timestamp'], results_df['window_mean'], 'g-', linewidth=1, label='Window Mean')
   
   # Highlight change points
   change_points = [1000, 2500, 4000]
   for cp in change_points:
       ax.axvline(cp, color='orange', linestyle='--', alpha=0.7)
   
   # Highlight anomalies
   anomaly_mask = results_df['is_anomaly']
   ax.scatter(results_df.loc[anomaly_mask, 'timestamp'], 
             results_df.loc[anomaly_mask, 'value'], 
             color='red', s=20, alpha=0.8, label='Anomalies')
   
   ax.set_xlabel('Time')
   ax.set_ylabel('Value')
   ax.set_title('Streaming Data with Real-time Statistics')
   ax.legend()
   
   # Confidence bands using streaming quantiles
   ax = axes[0, 1]
   
   ax.plot(results_df['timestamp'], results_df['value'], 'b-', alpha=0.6, linewidth=0.5)
   ax.plot(results_df['timestamp'], results_df['q50'], 'g-', linewidth=2, label='Streaming Median')
   
   # Confidence band using quantiles
   ax.fill_between(results_df['timestamp'], 
                   results_df['q25'], results_df['q75'],
                   alpha=0.3, color='green', label='IQR Band')
   
   # Highlight anomalies
   ax.scatter(results_df.loc[anomaly_mask, 'timestamp'], 
             results_df.loc[anomaly_mask, 'value'], 
             color='red', s=20, alpha=0.8, label='Anomalies')
   
   ax.set_xlabel('Time')
   ax.set_ylabel('Value')
   ax.set_title('Streaming Quantiles and Anomaly Detection')
   ax.legend()
   
   # Change point detection using CUSUM
   ax = axes[1, 0]
   
   ax.plot(results_df['timestamp'], results_df['cusum_pos'], 'r-', label='CUSUM+', linewidth=2)
   ax.plot(results_df['timestamp'], results_df['cusum_neg'], 'b-', label='CUSUM-', linewidth=2)
   ax.axhline(processor.change_threshold, color='orange', linestyle='--', 
             label='Threshold', alpha=0.7)
   
   # Highlight detected change points
   change_detected_mask = results_df['change_detected']
   if change_detected_mask.any():
       ax.scatter(results_df.loc[change_detected_mask, 'timestamp'], 
                 results_df.loc[change_detected_mask, 'cusum_pos'], 
                 color='red', s=30, marker='x', label='Detected Changes')
   
   ax.set_xlabel('Time')
   ax.set_ylabel('CUSUM Value')
   ax.set_title('Real-time Change Point Detection')
   ax.legend()
   
   # Forecasting performance
   ax = axes[1, 1]
   
   # Calculate forecast errors
   forecast_errors = results_df['value'] - results_df['forecast'].shift(1)
   forecast_errors = forecast_errors.dropna()
   
   ax.plot(results_df['timestamp'][1:], forecast_errors, 'purple', alpha=0.6, linewidth=0.5)
   ax.axhline(0, color='black', linestyle='-', alpha=0.5)
   ax.axhline(forecast_errors.std(), color='red', linestyle='--', alpha=0.7, label='+1 Std')
   ax.axhline(-forecast_errors.std(), color='red', linestyle='--', alpha=0.7, label='-1 Std')
   
   ax.set_xlabel('Time')
   ax.set_ylabel('Forecast Error')
   ax.set_title('Real-time Forecasting Errors')
   ax.legend()
   
   # Memory usage simulation
   ax = axes[2, 0]
   
   # Simulate memory usage (constant for streaming algorithms)
   memory_usage = processor.window_size * 8 / 1024  # KB, assuming 8 bytes per float
   
   ax.axhline(memory_usage, color='blue', linewidth=3, label='Streaming Algorithm')
   
   # Compare with batch algorithm memory usage
   batch_memory = results_df['timestamp'] * 8 / 1024  # Linear growth
   ax.plot(results_df['timestamp'], batch_memory, 'r-', linewidth=2, label='Batch Algorithm')
   
   ax.set_xlabel('Time')
   ax.set_ylabel('Memory Usage (KB)')
   ax.set_title('Memory Usage Comparison')
   ax.legend()
   ax.set_ylim(0, max(batch_memory.iloc[-1] * 1.1, memory_usage * 2))
   
   # Performance metrics over time
   ax = axes[2, 1]
   
   # Calculate rolling performance metrics
   window_size = 500
   
   rolling_mae = forecast_errors.abs().rolling(window_size).mean()
   rolling_mse = (forecast_errors**2).rolling(window_size).mean()
   
   ax.plot(results_df['timestamp'][1:window_size], rolling_mae[window_size:], 
           'b-', linewidth=2, label='Rolling MAE')
   ax2 = ax.twinx()
   ax2.plot(results_df['timestamp'][1:window_size], rolling_mse[window_size:], 
            'r-', linewidth=2, label='Rolling MSE')
   
   ax.set_xlabel('Time')
   ax.set_ylabel('MAE', color='blue')
   ax2.set_ylabel('MSE', color='red')
   ax.set_title('Rolling Forecast Performance')
   
   # Combine legends
   lines1, labels1 = ax.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
   
   plt.tight_layout()
   plt.show()
   
   # Performance analysis
   print("\nStreaming Algorithm Performance Analysis:")
   print("=" * 50)
   
   final_stats = processor.get_statistics()
   print(f"\nFinal Statistics:")
   print(f"  Samples processed: {final_stats['count']:,}")
   print(f"  Mean: {final_stats['mean']:.4f}")
   print(f"  Standard deviation: {final_stats['std']:.4f}")
   print(f"  Min: {final_stats['min']:.4f}")
   print(f"  Max: {final_stats['max']:.4f}")
   
   if 'q25' in final_stats:
       print(f"  25th percentile: {final_stats['q25']:.4f}")
       print(f"  50th percentile: {final_stats['q50']:.4f}")
       print(f"  75th percentile: {final_stats['q75']:.4f}")
   
   print(f"\nAnomaly Detection:")
   total_anomalies = sum(results_df['is_anomaly'])
   anomaly_rate = total_anomalies / len(results_df)
   print(f"  Total anomalies detected: {total_anomalies}")
   print(f"  Anomaly rate: {anomaly_rate:.4f} ({anomaly_rate*100:.2f}%)")
   
   print(f"\nChange Point Detection:")
   change_detections = sum(results_df['change_detected'])
   print(f"  Change points detected: {change_detections}")
   print(f"  True change points: {len(change_points)}")
   
   print(f"\nForecasting Performance:")
   mae = forecast_errors.abs().mean()
   mse = (forecast_errors**2).mean()
   rmse = np.sqrt(mse)
   
   print(f"  Mean Absolute Error: {mae:.4f}")
   print(f"  Mean Squared Error: {mse:.4f}")
   print(f"  Root Mean Squared Error: {rmse:.4f}")
   
   print(f"\nComputational Efficiency:")
   print(f"  Processing rate: {len(streaming_data)/processing_time:.0f} samples/second")
   print(f"  Memory usage: {memory_usage:.2f} KB (constant)")
   print(f"  Equivalent batch memory: {batch_memory.iloc[-1]:.2f} KB (final)")
   print(f"  Memory efficiency: {batch_memory.iloc[-1]/memory_usage:.1f}x reduction")
   
   print(f"\nAlgorithm Characteristics:")
   print("- O(1) space complexity per update")
   print("- O(1) time complexity per update") 
   print("- Single-pass processing")
   print("- Constant memory usage")
   print("- Real-time capability")
   
   # Benchmark against batch processing
   print(f"\nBatch vs Streaming Comparison:")
   
   # Simulate batch processing
   start_time = time.time()
   
   batch_data = np.array(streaming_data)
   batch_mean = np.mean(batch_data)
   batch_std = np.std(batch_data)
   batch_quantiles = np.percentile(batch_data, [25, 50, 75])
   
   batch_time = time.time() - start_time
   
   print(f"  Batch processing time: {batch_time:.6f}s")
   print(f"  Streaming processing time: {processing_time:.6f}s")
   print(f"  Time ratio: {processing_time/batch_time:.1f}x")
   
   print(f"\n  Accuracy Comparison:")
   print(f"    Mean - Batch: {batch_mean:.6f}, Streaming: {final_stats['mean']:.6f}")
   print(f"    Std - Batch: {batch_std:.6f}, Streaming: {final_stats['std']:.6f}")
   print(f"    Median - Batch: {batch_quantiles[1]:.6f}, Streaming: {final_stats.get('q50', 0):.6f}")
   
   accuracy_mean = abs(batch_mean - final_stats['mean']) / abs(batch_mean) if batch_mean != 0 else 0
   accuracy_std = abs(batch_std - final_stats['std']) / batch_std if batch_std != 0 else 0
   
   print(f"    Mean relative error: {accuracy_mean:.6f}")
   print(f"    Std relative error: {accuracy_std:.6f}")
   ```

3. **Connection to Exercise 14.4:**
   This comprehensive streaming algorithm implementation provides the foundation for Exercise 14.4, demonstrating how to process large-scale time series data efficiently with constant memory usage and real-time capability.

### Worked Example 4: Performance Benchmarking and Optimization
**Context:** This example demonstrates systematic approach to benchmarking and optimizing time series algorithms, preparing students for comprehensive performance analysis.

1. **Theoretical Background:**
   - Systematic benchmarking requires controlling for multiple variables
   - Different optimization strategies suit different algorithmic patterns
   - Profiling helps identify bottlenecks and guide optimization efforts

2. **Example:**
   Let's implement a comprehensive benchmarking and optimization framework.
   
   **Step 1:** Benchmarking framework with profiling
   ```python
   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from memory_profiler import profile
   import cProfile
   import pstats
   from functools import wraps
   import pandas as pd
   
   class TimeSeriesBenchmark:
       def __init__(self):
           self.results = {}
           
       def time_function(self, func):
           """Decorator for timing function execution"""
           @wraps(func)
           def wrapper(*args, **kwargs):
               start_time = time.perf_counter()
               result = func(*args, **kwargs)
               end_time = time.perf_counter()
               return result, end_time - start_time
           return wrapper
       
       def benchmark_algorithm(self, algorithm_func, data_sizes, n_trials=5, **kwargs):
           """Benchmark an algorithm across different data sizes"""
           
           results = {
               'data_size': [],
               'mean_time': [],
               'std_time': [],
               'min_time': [],
               'max_time': [],
               'memory_peak': []
           }
           
           for size in data_sizes:
               print(f"Benchmarking size: {size:,}")
               
               times = []
               
               for trial in range(n_trials):
                   # Generate test data
                   np.random.seed(42 + trial)  # Different seed per trial
                   test_data = np.random.randn(size)
                   
                   # Time the algorithm
                   start_time = time.perf_counter()
                   try:
                       result = algorithm_func(test_data, **kwargs)
                       end_time = time.perf_counter()
                       times.append(end_time - start_time)
                   except Exception as e:
                       print(f"Error in trial {trial} for size {size}: {e}")
                       times.append(float('inf'))
               
               # Calculate statistics
               times = [t for t in times if not np.isinf(t)]
               
               if times:
                   results['data_size'].append(size)
                   results['mean_time'].append(np.mean(times))
                   results['std_time'].append(np.std(times))
                   results['min_time'].append(np.min(times))
                   results['max_time'].append(np.max(times))
                   results['memory_peak'].append(0)  # Placeholder
               
           return pd.DataFrame(results)
       
       def profile_algorithm(self, algorithm_func, data_size=10000, **kwargs):
           """Profile an algorithm to identify bottlenecks"""
           
           # Generate test data
           np.random.seed(42)
           test_data = np.random.randn(data_size)
           
           # Create wrapper function for profiling
           def run_algorithm():
               return algorithm_func(test_data, **kwargs)
           
           # Profile execution
           pr = cProfile.Profile()
           pr.enable()
           
           result = run_algorithm()
           
           pr.disable()
           
           # Get profile statistics
           stats = pstats.Stats(pr)
           stats.sort_stats('cumulative')
           
           return stats, result
       
       def compare_implementations(self, implementations, data_size=10000, n_trials=5):
           """Compare multiple implementations of the same algorithm"""
           
           results = {
               'implementation': [],
               'mean_time': [],
               'std_time': [],
               'speedup': [],
               'memory_usage': []
           }
           
           # Generate test data once
           np.random.seed(42)
           test_data = np.random.randn(data_size)
           
           baseline_time = None
           
           for name, func in implementations.items():
               print(f"Benchmarking {name}...")
               
               times = []
               
               for trial in range(n_trials):
                   start_time = time.perf_counter()
                   try:
                       result = func(test_data.copy())
                       end_time = time.perf_counter()
                       times.append(end_time - start_time)
                   except Exception as e:
                       print(f"Error in {name}: {e}")
                       times.append(float('inf'))
               
               times = [t for t in times if not np.isinf(t)]
               
               if times:
                   mean_time = np.mean(times)
                   std_time = np.std(times)
                   
                   if baseline_time is None:
                       baseline_time = mean_time
                       speedup = 1.0
                   else:
                       speedup = baseline_time / mean_time
                   
                   results['implementation'].append(name)
                   results['mean_time'].append(mean_time)
                   results['std_time'].append(std_time)
                   results['speedup'].append(speedup)
                   results['memory_usage'].append(0)  # Placeholder
           
           return pd.DataFrame(results)
   
   # Define algorithms to benchmark
   def naive_autocorrelation(data, max_lag=100):
       """Naive O(n*k) autocorrelation implementation"""
       n = len(data)
       mean = np.mean(data)
       var = np.var(data)
       
       autocorr = []
       for lag in range(min(max_lag, n)):
           if lag == 0:
               autocorr.append(1.0)
           else:
               numerator = np.sum((data[:-lag] - mean) * (data[lag:] - mean))
               autocorr.append(numerator / ((n - lag) * var))
       
       return np.array(autocorr)
   
   def fft_autocorrelation(data, max_lag=100):
       """FFT-based O(n log n) autocorrelation"""
       n = len(data)
       
       # Zero-pad to avoid circular correlation
       padded_data = np.zeros(2 * n)
       padded_data[:n] = data - np.mean(data)
       
       # FFT-based correlation
       fft_data = np.fft.fft(padded_data)
       autocorr_full = np.fft.ifft(fft_data * np.conj(fft_data)).real
       
       # Normalize
       autocorr_full = autocorr_full / autocorr_full[0]
       
       return autocorr_full[:min(max_lag, n)]
   
   def optimized_autocorrelation(data, max_lag=100):
       """Optimized implementation using NumPy vectorization"""
       n = len(data)
       mean = np.mean(data)
       var = np.var(data)
       
       centered = data - mean
       
       # Vectorized computation
       autocorr = []
       for lag in range(min(max_lag, n)):
           if lag == 0:
               autocorr.append(1.0)
           else:
               corr = np.correlate(centered, centered[:-lag], mode='valid')[0]
               autocorr.append(corr / ((n - lag) * var))
       
       return np.array(autocorr)
   
   def numpy_autocorrelation(data, max_lag=100):
       """Using NumPy's built-in correlation"""
       from scipy.signal import correlate
       
       centered = data - np.mean(data)
       
       # Full correlation
       full_corr = correlate(centered, centered, mode='full')
       mid = len(full_corr) // 2
       
       # Take positive lags and normalize
       autocorr = full_corr[mid:mid + min(max_lag, len(data))]
       autocorr = autocorr / autocorr[0]
       
       return autocorr
   ```
   
   **Step 2:** Comprehensive benchmarking analysis
   ```python
   # Initialize benchmark system
   benchmark = TimeSeriesBenchmark()
   
   # Define implementations to compare
   autocorr_implementations = {
       'Naive': naive_autocorrelation,
       'FFT-based': fft_autocorrelation,
       'Optimized': optimized_autocorrelation,
       'NumPy/SciPy': numpy_autocorrelation
   }
   
   # Benchmark scaling behavior
   data_sizes = [100, 500, 1000, 2000, 5000, 10000]
   
   scaling_results = {}
   
   for name, func in autocorr_implementations.items():
       print(f"\nBenchmarking scaling for {name}...")
       scaling_results[name] = benchmark.benchmark_algorithm(
           func, data_sizes, n_trials=3, max_lag=50
       )
   
   # Compare implementations at fixed size
   comparison_results = benchmark.compare_implementations(
       autocorr_implementations, data_size=5000, n_trials=5
   )
   
   # Profile the naive implementation to identify bottlenecks
   print("\nProfiling naive implementation...")
   stats, _ = benchmark.profile_algorithm(naive_autocorrelation, data_size=2000, max_lag=100)
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(2, 3, figsize=(18, 12))
   
   # Scaling comparison
   ax = axes[0, 0]
   
   for name, results in scaling_results.items():
       if len(results) > 0:
           ax.loglog(results['data_size'], results['mean_time'], 
                    'o-', label=name, alpha=0.8, markersize=4)
           
           # Add error bars
           ax.errorbar(results['data_size'], results['mean_time'], 
                      yerr=results['std_time'], alpha=0.5, capsize=3)
   
   # Add theoretical complexity lines
   sizes_smooth = np.logspace(np.log10(min(data_sizes)), np.log10(max(data_sizes)), 100)
   
   # O(n²) reference
   n_squared = (sizes_smooth / 1000) ** 2 / 1e6
   ax.loglog(sizes_smooth, n_squared, '--', alpha=0.5, color='red', label='O(n²)')
   
   # O(n log n) reference
   n_log_n = sizes_smooth * np.log2(sizes_smooth) / 1e6
   ax.loglog(sizes_smooth, n_log_n, '--', alpha=0.5, color='green', label='O(n log n)')
   
   ax.set_xlabel('Data Size')
   ax.set_ylabel('Time (seconds)')
   ax.set_title('Scaling Behavior Comparison')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Implementation comparison
   ax = axes[0, 1]
   
   if len(comparison_results) > 0:
       bars = ax.bar(comparison_results['implementation'], 
                    comparison_results['mean_time'], 
                    yerr=comparison_results['std_time'],
                    alpha=0.8, capsize=5)
       
       # Color bars by performance
       times = comparison_results['mean_time'].values
       max_time = max(times)
       
       for bar, time_val in zip(bars, times):
           if time_val < max_time * 0.2:
               bar.set_color('green')
           elif time_val < max_time * 0.5:
               bar.set_color('orange')
           else:
               bar.set_color('red')
       
       ax.set_ylabel('Time (seconds)')
       ax.set_title('Implementation Comparison (5K samples)')
       ax.tick_params(axis='x', rotation=45)
   
   # Speedup comparison
   ax = axes[0, 2]
   
   if len(comparison_results) > 0:
       speedups = comparison_results['speedup'].values
       implementations = comparison_results['implementation'].values
       
       bars = ax.bar(implementations, speedups, alpha=0.8)
       ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
       
       # Annotate bars with speedup values
       for bar, speedup in zip(bars, speedups):
           height = bar.get_height()
           ax.text(bar.get_x() + bar.get_width()/2., height,
                  f'{speedup:.1f}x', ha='center', va='bottom')
       
       ax.set_ylabel('Speedup Factor')
       ax.set_title('Relative Performance Speedup')
       ax.tick_params(axis='x', rotation=45)
       ax.legend()
   
   # Memory usage simulation
   ax = axes[1, 0]
   
   # Simulate memory usage patterns for different implementations
   sizes = np.array(data_sizes)
   
   naive_memory = sizes * 8 + 100 * 8  # O(n) + autocorr storage
   fft_memory = sizes * 8 * 2 + sizes * 16  # Padded + complex FFT
   optimized_memory = sizes * 8 + 100 * 8  # Similar to naive
   
   ax.plot(sizes, naive_memory / 1024, 'o-', label='Naive', alpha=0.8)
   ax.plot(sizes, fft_memory / 1024, 's-', label='FFT-based', alpha=0.8)  
   ax.plot(sizes, optimized_memory / 1024, '^-', label='Optimized', alpha=0.8)
   
   ax.set_xlabel('Data Size')
   ax.set_ylabel('Memory Usage (KB)')
   ax.set_title('Memory Usage Comparison')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # Efficiency analysis (time vs accuracy)
   ax = axes[1, 1]
   
   # Generate reference autocorrelation for accuracy comparison
   np.random.seed(42)
   test_data = np.random.randn(1000)
   reference = naive_autocorrelation(test_data, max_lag=50)
   
   accuracies = []
   times = []
   names = []
   
   for name, func in autocorr_implementations.items():
       if name != 'Naive':  # Skip reference
           try:
               start_time = time.perf_counter()
               result = func(test_data, max_lag=50)
               end_time = time.perf_counter()
               
               # Calculate accuracy (correlation with reference)
               min_len = min(len(reference), len(result))
               accuracy = np.corrcoef(reference[:min_len], result[:min_len])[0, 1]
               
               accuracies.append(accuracy)
               times.append(end_time - start_time)
               names.append(name)
               
           except Exception as e:
               print(f"Error testing {name}: {e}")
   
   if accuracies and times:
       scatter = ax.scatter(times, accuracies, s=100, alpha=0.8)
       
       for i, name in enumerate(names):
           ax.annotate(name, (times[i], accuracies[i]), 
                      xytext=(5, 5), textcoords='offset points')
       
       ax.set_xlabel('Time (seconds)')
       ax.set_ylabel('Accuracy (correlation with reference)')
       ax.set_title('Efficiency vs Accuracy Trade-off')
       ax.grid(True, alpha=0.3)
   
   # Performance over different lag values
   ax = axes[1, 2]
   
   lag_values = [10, 25, 50, 100, 200]
   np.random.seed(42)
   test_data = np.random.randn(2000)
   
   lag_performance = {}
   
   for name, func in autocorr_implementations.items():
       times = []
       
       for max_lag in lag_values:
           start_time = time.perf_counter()
           try:
               result = func(test_data, max_lag=max_lag)
               end_time = time.perf_counter()
               times.append(end_time - start_time)
           except:
               times.append(float('inf'))
       
       lag_performance[name] = times
   
   for name, times in lag_performance.items():
       valid_times = [t for t in times if not np.isinf(t)]
       valid_lags = [lag_values[i] for i, t in enumerate(times) if not np.isinf(t)]
       
       if valid_times:
           ax.plot(valid_lags, valid_times, 'o-', label=name, alpha=0.8)
   
   ax.set_xlabel('Maximum Lag')
   ax.set_ylabel('Time (seconds)')
   ax.set_title('Performance vs Lag Parameter')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()
   
   # Print comprehensive analysis
   print("\nPerformance Analysis Summary:")
   print("=" * 50)
   
   print(f"\nImplementation Comparison (5K samples):")
   if len(comparison_results) > 0:
       for _, row in comparison_results.iterrows():
           name = row['implementation']
           mean_time = row['mean_time']
           speedup = row['speedup']
           
           print(f"  {name:15s}: {mean_time:.6f}s ({speedup:.1f}x speedup)")
   
   print(f"\nScaling Analysis:")
   for name, results in scaling_results.items():
       if len(results) >= 2:
           # Estimate complexity from scaling
           sizes = results['data_size'].values
           times = results['mean_time'].values
           
           # Fit to power law: T = a * N^b
           log_sizes = np.log(sizes)
           log_times = np.log(times)
           
           if len(log_sizes) > 1:
               coeffs = np.polyfit(log_sizes, log_times, 1)
               complexity_exp = coeffs[0]
               
               print(f"  {name:15s}: O(N^{complexity_exp:.2f})")
   
   print(f"\nOptimization Recommendations:")
   
   if len(comparison_results) > 0:
       best_impl = comparison_results.loc[comparison_results['mean_time'].idxmin(), 'implementation']
       worst_impl = comparison_results.loc[comparison_results['mean_time'].idxmax(), 'implementation']
       
       best_time = comparison_results['mean_time'].min()
       worst_time = comparison_results['mean_time'].max()
       
       print(f"- Best performing: {best_impl} ({best_time:.6f}s)")
       print(f"- Worst performing: {worst_impl} ({worst_time:.6f}s)")
       print(f"- Performance gap: {worst_time/best_time:.1f}x")
   
   print(f"\n- Use FFT-based methods for large datasets")
   print(f"- Consider memory vs time trade-offs")
   print(f"- Vectorized operations outperform loops")
   print(f"- Library implementations are usually optimal")
   
   # Advanced profiling demonstration
   print(f"\nDetailed Profiling (Naive Implementation):")
   print("-" * 30)
   
   # Show top 10 most time-consuming functions
   stats.print_stats(10)
   
   # Memory profiling example
   print(f"\nMemory Usage Optimization Tips:")
   print("- Use in-place operations when possible")
   print("- Avoid unnecessary data copies")
   print("- Consider data type optimization (float32 vs float64)")
   print("- Use generators for large datasets")
   print("- Profile memory usage in production scenarios")
   ```

3. **Connection to Chapter Exercises:**
   This comprehensive benchmarking framework provides the tools needed for systematic performance analysis, demonstrating how to identify bottlenecks, compare implementations, and make data-driven optimization decisions - essential skills for high-performance time series computing.

These worked examples provide hands-on experience with the key computational efficiency concepts covered in Chapter 14, preparing students to tackle performance optimization challenges while demonstrating practical approaches to algorithm design, benchmarking, and optimization in time series analysis.