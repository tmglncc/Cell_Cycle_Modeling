import numpy as np
import pysindy_local as ps
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RBF, WhiteKernel, RationalQuadratic, DotProduct, ConstantKernel
from ModelPlots import ModelPlots
from ModelCalibration import ModelCalibration
from ModelSelection import ModelSelection
from DataDenoising import DataDenoising

def is_new_model(model_set, model, n_vars, precision):
	for model_element in model_set:
		flag = True
		for i in range(n_vars):
			if model_element.equations(precision = precision)[i] != model.equations(precision = precision)[i]:
				flag = False
				break

		if flag:
			return False

	return True

def true_model(X, t, r01, r20, r12, d, rA):
	G0G1, S, G2M, apoptotic = X
	dXdt = [-r01*G0G1 + 2.0*r20*G2M - d*G0G1,
		r01*G0G1 - r12*S - d*S,
		r12*S - r20*G2M - d*G2M,
		d*G0G1 + d*S + d*G2M - rA*apoptotic
	]
	return dXdt

experiment_id = 0

# Train and test data parameters
step = 1

# Method parameters
fd_order = 2
poly_degrees = range(1, 3)
fourier_nfreqs = range(1, 2)
optimizer_method = "STLSQ+SA"
precision = 3

plot_sse = True
plot_sse_correlation = False
plot_relative_error = True
plot_Ftest = True
plot_qoi = True
plot_musig = False
plot_simulation = False
plot_derivative = False
calibration_mode = "DE"

stlsq_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

# Read train data
data = np.genfromtxt("data.csv", dtype = float, delimiter = ',', names = True)
t = data["times"][::step]
X = np.stack((data["G0G1_phase_mean"][::step], data["S_phase_mean"][::step], data["G2M_phase_mean"][::step], data["apoptotic_mean"][::step]), axis = -1)

data_std = np.genfromtxt("data_std.csv", dtype = float, delimiter = ',', names = True)
X_std = np.stack((data_std["G0G1_phase_std"][::step], data_std["S_phase_std"][::step], data_std["G2M_phase_std"][::step], data_std["apoptotic_std"][::step]), axis = -1)

X0 = X[0, :]
t_steps = len(t)

# Read test data
data_test = np.genfromtxt("data.csv", dtype = float, delimiter = ',', names = True)
t_test = data["times"][::step]
X_test = np.stack((data["G0G1_phase_mean"][::step], data["S_phase_mean"][::step], data["G2M_phase_mean"][::step], data["apoptotic_mean"][::step]), axis = -1)

X0_test = X_test[0, :]

# Solve true model
r01 = 0.003236246
r20 = 0.003333333
r12 = 0.002083333
d = 0.000053167
rA = 0.001937984
true_solution = odeint(true_model, X0_test, t_test, args=(r01, r20, r12, d, rA))

model_set = []
for poly_degree in poly_degrees:
	for fourier_nfreq in fourier_nfreqs:
		for stlsq_alpha in stlsq_alphas:
			experiment_id += 1
			print("Experimento " + str(experiment_id) 
				+ ": Grau = " + str(poly_degree) 
				+ ", Frequência = " + str(fourier_nfreq) 
				+ ", alpha = " + str(stlsq_alpha) + "\n"
			)

			# Define method properties
			differentiation_method = ps.FiniteDifference(order = fd_order)
			# differentiation_method = ps.SmoothedFiniteDifference()
			feature_library = ps.PolynomialLibrary(degree = poly_degree) # + ps.FourierLibrary(n_frequencies = fourier_nfreq)
			optimizer = ps.STLSQ(
				alpha = stlsq_alpha,
				fit_intercept = False,
				verbose = True,
				window = 3,
				epsilon = 3.0e-7,
				time = t,
				sa_times = np.array([3600.0]),
				non_physical_features = [['1', 'S', 'G0G1 S', 'S^2', 'S G2M', 'apoptotic', 'G0G1 apoptotic', 'S apoptotic', 'G2M apoptotic', 'apoptotic^2'],
					['1', 'G2M', 'G0G1 G2M', 'S G2M', 'G2M^2', 'apoptotic', 'G0G1 apoptotic', 'S apoptotic', 'G2M apoptotic', 'apoptotic^2'],
					['1', 'G0G1', 'G0G1^2', 'G0G1 S', 'G0G1 G2M', 'apoptotic', 'G0G1 apoptotic', 'S apoptotic', 'G2M apoptotic', 'apoptotic^2'],
					['1']
				]
			)
			# Feature names = ['1', 'G0G1', 'S', 'G2M', 'apoptotic', 'G0G1^2', 'G0G1 S', 'G0G1 G2M', 'G0G1 apoptotic', 'S^2', 'S G2M', 'S apoptotic', 'G2M^2', 'G2M apoptotic', 'apoptotic^2']

			# Compute sparse regression
			model = ps.SINDy(
				differentiation_method = differentiation_method,
				feature_library = feature_library,
				optimizer = optimizer,
				feature_names = ["G0G1", "S", "G2M", "apoptotic"]
			)
			model.fit(X, t = t)
			model.print(precision = precision)
			print("\n")

			# Generate model plots
			mp = ModelPlots(model, optimizer_method, experiment_id)
			if plot_sse:
				mp.plot_sse()
			if plot_sse_correlation:
				mp.plot_sse_correlation()
			if plot_relative_error:
				mp.plot_relative_error()
			if plot_Ftest:
				mp.plot_Ftest()
			if plot_qoi:
				mp.plot_qoi()
			if plot_musig:
				mp.plot_musig()
			if plot_simulation:
				mp.plot_simulation(X, t, X0, precision = precision)
			if plot_derivative:
				mp.plot_derivative(X, t)

			# Add model to the set of models
			if not model_set or is_new_model(model_set, model, len(model.feature_names), precision):
				model_set.append(model)

# Compute number of terms
ms = ModelSelection(model_set, t_steps)
ms.compute_k()

for model_id, model in enumerate(model_set):
	print("Modelo " + str(model_id+1) + "\n")
	model.print(precision = precision)
	print("\n")

	# Simulate with another initial condition
	try:
		if calibration_mode is None:
			simulation = model.simulate(X0_test, t = t_test)
		elif calibration_mode == "LM":
			mc = ModelCalibration(model, model_id, X_test, t, X0_test, 0)
			mc.levenberg_marquardt(X_std, normalize = True)
			model.print(precision = precision)
			print("\n")

			simulation = model.simulate(X0_test, t = t_test)
		elif calibration_mode == "DE":
			mc = ModelCalibration(model, model_id, X_test, t, X0_test, 0)
			mc.differential_evolution(normalize = True)
			model.print(precision = precision)
			print("\n")

			simulation = model.simulate(X0_test, t = t_test)
		elif calibration_mode == "Bayes":
			mc = ModelCalibration(model, model_id, X_test, t, X0_test, 0)
			mc.bayesian_calibration()
			mc.traceplot()
			mc.plot_posterior()
			mc.plot_pair()
			X0_test = mc.summary()
			print("\n")
			model.print(precision = precision)
			print("\n")

			simulation, simulation_min, simulation_max = mc.get_simulation()
	except:
		print("Modelo " + str(model_id+1) + " não pode ser simulado ou recalibrado" + "\n")
		continue

	# Generate figures
	for std_factor in range(1, 2):
		fig, ax1 = plt.subplots(1, 1, figsize = (10.54, 5.92), dpi = 300)
		ax2 = ax1.twinx()
		ax1.plot(t_test, X_test[:,0], "ko", label = r"Data G$_0$ G$_1$$(t)$", alpha = 0.3, markersize = 3)
		ax1.plot(t_test, X_test[:,1], "k^", label = r"Data S$(t)$", alpha = 0.3, markersize = 3)
		ax1.plot(t_test, X_test[:,2], "ks", label = r"Data G$_2$ M$(t)$", alpha = 0.3, markersize = 3)
		# ax2.plot(t_test, X_test[:,3], "kD", label = r"Data $A(t)$", alpha = 0.3, markersize = 3)
		# ax1.errorbar(t_test, X_test[:,0], yerr = std_factor*X_std[:,0], fmt = 'ko', label = r"Data G$_0$ G$_1$$(t)$", capsize = 2.0, alpha = 0.3, markersize = 3)
		# ax1.errorbar(t_test, X_test[:,1], yerr = std_factor*X_std[:,1], fmt = 'k^', label = r"Data S$(t)$", capsize = 2.0, alpha = 0.3, markersize = 3)
		# ax1.errorbar(t_test, X_test[:,2], yerr = std_factor*X_std[:,2], fmt = 'ks', label = r"Data G$_2$ M$(t)$", capsize = 2.0, alpha = 0.3, markersize = 3)
		ax2.errorbar(t_test, X_test[:,3], yerr = std_factor*X_std[:,3], fmt = 'kD', label = r"Data $A(t)$", capsize = 2.0, alpha = 0.3, markersize = 3)
		ax1.plot(t_test, true_solution[:,0], "b:", label = r"True G$_0$ G$_1$$(t)$", alpha = 1.0, linewidth = 1)
		ax1.plot(t_test, true_solution[:,1], "g:", label = r"True S$(t)$", alpha = 1.0, linewidth = 1)
		ax1.plot(t_test, true_solution[:,2], "r:", label = r"True G$_2$ M$(t)$", alpha = 1.0, linewidth = 1)
		ax2.plot(t_test, true_solution[:,3], "y:", label = r"True $A(t)$", alpha = 1.0, linewidth = 1)
		ax1.plot(t_test, simulation[:,0], "b", label = r"SINDy-SA G$_0$ G$_1$$(t)$", alpha = 1.0, linewidth = 1)
		ax1.plot(t_test, simulation[:,1], "g", label = r"SINDy-SA S$(t)$", alpha = 1.0, linewidth = 1)
		ax1.plot(t_test, simulation[:,2], "r", label = r"SINDy-SA G$_2$ M$(t)$", alpha = 1.0, linewidth = 1)
		ax2.plot(t_test, simulation[:,3], "y", label = r"SINDy-SA $A(t)$", alpha = 1.0, linewidth = 1)
		if calibration_mode == "Bayes":
			ax1.fill_between(t, simulation_min[:,0], simulation_max[:,0], color = "b", alpha = 0.4)
			ax1.fill_between(t, simulation_min[:,1], simulation_max[:,1], color = "g", alpha = 0.4)
			ax1.fill_between(t, simulation_min[:,2], simulation_max[:,2], color = "r", alpha = 0.4)
			ax2.fill_between(t, simulation_min[:,3], simulation_max[:,3], color = "y", alpha = 0.4)
		ax1.set_xlabel(r"Time $t$")
		ax1.set_ylabel(r"G$_0$ G$_1$$(t)$, S$(t)$, G$_2$ M$(t)$", color = 'k')
		ax2.set_ylabel(r"$A(t)$", color = 'k')
		ax1.legend(loc = 2)
		ax2.legend(loc = 4)
		plt.savefig(os.path.join("output", "model" + str(model_id+1) + "_ic0_std" + str(std_factor) + ".pdf"), bbox_inches = 'tight')
		plt.close()

	# Compute SSE
	sse = ms.compute_SSE(np.copy(X_test.reshape(simulation.shape)), np.copy(simulation), normalize = True)

	# Set mean SSE to the model
	ms.set_model_SSE(model_id, sse)

# Compute AIC and AICc
best_AIC_model = ms.compute_AIC()
best_AICc_model = ms.compute_AICc()
best_BIC_model = ms.compute_BIC()

# Get best model
print("Melhor modelo AIC = " + str(best_AIC_model+1) + "\n")
print("Melhor modelo AICc = " + str(best_AICc_model+1) + "\n")
print("Melhor modelo BIC = " + str(best_BIC_model+1) + "\n")

# Write results
ms.write_output()
ms.write_AICc_weights()
ms.write_pareto_curve(optimizer_method)

# # Compute cumulative sum of squared errors between data and true model
# csse = np.zeros(true_solution.shape)
# csse[:,0] = ms.compute_CSSE(true_solution[:,0], X_test[:,0])
# csse[:,1] = ms.compute_CSSE(true_solution[:,1], X_test[:,1])
# print("SSE between data and true model = " + str(csse[-1,:]) + "\n")

# fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
# ax.plot(t_test, csse[:,0], "b", label = r"Ki67$^{-}(t)$", alpha = 1.0, linewidth = 1)
# ax.plot(t_test, csse[:,1], "g", label = r"Ki67$^{+}(t)$", alpha = 1.0, linewidth = 1)
# ax.set(xlabel = r"Time $t$", ylabel = "Cumulative SSE")
# ax.legend()
# plt.savefig(os.path.join("output", "csse.png"), bbox_inches = 'tight')
# plt.close()

# # Compute normalized cumulative sum of squared errors between data and true model
# ncsse = np.zeros(true_solution.shape)
# ncsse[:,0] = ms.compute_NCSSE(true_solution[:,0], X_test[:,0])
# ncsse[:,1] = ms.compute_NCSSE(true_solution[:,1], X_test[:,1])
# print("Normalized SSE between data and true model = " + str(ncsse[-1,:]) + "\n")

# fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
# ax.plot(t_test, ncsse[:,0], "b", label = r"Ki67$^{-}(t)$", alpha = 1.0, linewidth = 1)
# ax.plot(t_test, ncsse[:,1], "g", label = r"Ki67$^{+}(t)$", alpha = 1.0, linewidth = 1)
# ax.set(xlabel = r"Time $t$", ylabel = "Normalized Cumulative SSE")
# ax.legend()
# plt.savefig(os.path.join("output", "ncsse.png"), bbox_inches = 'tight')
# plt.close()

# output_array = np.column_stack((t_test.flatten(), ncsse[:,0].flatten(), ncsse[:,1].flatten()))
# np.savetxt(os.path.join("output", "ncsse.csv"), output_array, delimiter=',', fmt='%.8f',
# 	header="times, Ki67_negative_error, Ki67_positive_error", comments='')
