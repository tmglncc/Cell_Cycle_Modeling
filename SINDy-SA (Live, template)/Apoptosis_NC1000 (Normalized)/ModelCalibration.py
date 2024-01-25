import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pymc3 as pm
import time
import arviz as az
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.stats import gaussian_kde
from tqdm import tqdm

class ModelCalibration:
	def __init__(self, model = None, model_id = 0, X = None, t = None, X0 = None, init_cond_id = 0):
		self.model = model
		self.model_id = model_id
		self.X = X
		self.t = t
		self.X0 = X0
		self.init_cond_id = init_cond_id

	def __input_fmt(self, string):
		return string.replace(" ", "*").replace("^", "**").replace("cos", "np.cos").replace("sin", "np.sin").replace("log", "np.log")

	# def __create_model_func(self, param_expression, var_expression, model_expression, n_vars):
	# 	function = """def func(X0, normalization_factor, target):
	# 		import numpy as np
	# 		from scipy.integrate import odeint
	# 		def model(X, t, """ + param_expression + """):
	# 			""" + var_expression + """ = X
	# 			dXdt = """ + (model_expression if n_vars == 1 else """[""" + model_expression + """]""") + """
	# 			return dXdt

	# 		def subfunc(time, """ + param_expression + """):
	# 			solution = odeint(model, X0, time, args = (""" + param_expression +"""))
	# 			return solution[:, target]/normalization_factor[target]

	# 		return subfunc
	# 	"""
	# 	return function

	def __create_model_func(self, param_expression, var_expression, model_expression, n_vars):
		function = """def func(X0, normalization_factor):
			import numpy as np
			from scipy.integrate import odeint
			def model(X, t, """ + param_expression + """):
				""" + var_expression + """ = X
				dXdt = """ + (model_expression if n_vars == 1 else """[""" + model_expression + """]""") + """
				return dXdt

			def subfunc(time, """ + param_expression + """):
				solution = odeint(model, X0, time, args = (""" + param_expression +"""))
				solution_norm = np.full(solution.shape, normalization_factor)
				solution /= solution_norm
				return solution.flatten()

			return subfunc
		"""
		return function

	def __create_ssr_func(self, param_expression, var_expression, model_expression, n_vars):
		function = """def func(X0, normalization_factor):
			import numpy as np
			from scipy.integrate import odeint
			def model(X, t, """ + param_expression + """):
				""" + var_expression + """ = X
				dXdt = """ + (model_expression if n_vars == 1 else """[""" + model_expression + """]""") + """
				return dXdt

			def sum_of_squared_residuals(params, t, X):
				""" + param_expression + """ = params
				X_pred = odeint(model, X0, t, args = (""" + param_expression +"""))
				
				residuals = X - X_pred
				residuals_norm = np.full(residuals.shape, normalization_factor)
				residuals /= residuals_norm

				ssr = np.sum(residuals**2.0)
				return ssr

			return sum_of_squared_residuals
		"""
		return function

	def __create_model(self, param_expression, var_expression, model_expression, n_vars):
		function = """import numpy as np
def model(X, t, """ + param_expression + """):
	""" + var_expression + """ = X
	dXdt = """ + (model_expression if n_vars == 1 else """[""" + model_expression + """]""") + """
	return dXdt
		"""
		return function

	def __create_model_wrapper(self, itypes_expression, param_expression, init_cond_expression, n_vars):
		function = """import theano
import theano.tensor as tt
from scipy.integrate import odeint

THEANO_FLAGS = "optimizer=fast_compile"

@theano.compile.ops.as_op(
	itypes = [tt.dvector, """ + itypes_expression + """],
	otypes = [tt.dmatrix]
)
def model_wrapper(time, """ + param_expression + """ """ + init_cond_expression + """):
	return odeint(model, """ + (init_cond_expression if n_vars == 1 else """[""" + init_cond_expression + """],""") + """ time, args = (""" + param_expression +"""))
"""
		return function

	def __create_smc_code(self, param_names, bounds, sd_upper_bound, sd_shape, param_expression, init_cond_expression, draws, seed):
		code = """with pm.Model() as model_smc:
			"""
		for i, param_name in enumerate(param_names):
			code += param_name + """ = pm.Uniform(
				'""" + param_name + """',
				lower = """ + str(bounds[i][0]) + """,
				upper = """ + str(bounds[i][1]) + """,
			)
			"""
		code += """standard_deviation = pm.Uniform(
				'std_deviation',
				lower = 0,
				upper = """ + str(sd_upper_bound) + """,
				shape = """ + str(sd_shape) + """,
			)
			"""
		code += """time_calibration = pm.Data('time', self.t)
			"""
		code += """fitting_model = pm.Deterministic(
				'model',
				wrapper['model_wrapper'](time_calibration, """ + param_expression + """ """ + init_cond_expression + """),
			)
			"""
		code += """likelihood_model = pm.Normal(
				'likelihood_model',
				mu = fitting_model,
				sigma = standard_deviation,
				observed = self.X
			)
			"""
		code += """self.trace_calibration = pm.sample_smc(
				draws = """ + str(draws) + """,
				n_steps = 25,
				# parallel = True,
				# cores = 4,
				# progressbar = True,
				random_seed = """ + str(seed) + """
			)
			"""
		return code

	def __scalar_rv_mvp_estimation(self, rv_realization_values: np.ndarray) -> np.ndarray:
		num_of_realizations = len(rv_realization_values)
		kernel = gaussian_kde(rv_realization_values)
		equally_spaced_samples = np.linspace(
			rv_realization_values.min(),
			rv_realization_values.max(),
			num_of_realizations
		)
		kde = kernel(equally_spaced_samples)
		kde_max_index = np.argmax(kde)
		rv_mpv_value = equally_spaced_samples[kde_max_index]
		return rv_mpv_value

	def __calculate_rv_posterior_mpv(self, pm_trace, variable_names: list) -> dict:
		rv_mpv_values_dict = dict()
		progress_bar = tqdm(variable_names)
		for variable in progress_bar:
			progress_bar.set_description(f"Calulating MPV from KDE for {variable}")
			rv_realization_values = pm_trace[f"{variable}"]

			try:
				num_of_dimensions = rv_realization_values.shape[1]
			except IndexError:
				num_of_dimensions = 0

			if num_of_dimensions == 0:
				rv_mpv_value = self.__scalar_rv_mvp_estimation(rv_realization_values)
				rv_mpv_values_dict[f"{variable}"] = rv_mpv_value
			else:
				for dimension in range(num_of_dimensions):
					variable_name_decomposed = f"{variable}[{dimension}]"
					rv_realization_values_decomposed = np.array(rv_realization_values[:, dimension])
					rv_mpv_value = self.__scalar_rv_mvp_estimation(rv_realization_values_decomposed)
					rv_mpv_values_dict[f"{variable_name_decomposed}"] = rv_mpv_value

		return rv_mpv_values_dict

	def __add_mpv_to_summary(self, arviz_summary: pd.DataFrame, rv_modes_dict: dict) -> pd.DataFrame:
		new_arviz_summary = arviz_summary.copy()
		variable_names = list(rv_modes_dict.keys())
		rv_mode_values = list(rv_modes_dict.values())
		new_arviz_summary["mpv"] = pd.Series(data=rv_mode_values, index=variable_names)
		return new_arviz_summary

	def levenberg_marquardt(self, X_std, normalize = False, number_of_restarts = 50, bounds_perc = 0.5, seed = 7):
		print("*** Using Levenberg-Marquardt algorithm ***\n")

		ind = self.model.coefficients() != 0.0
		param_names = np.array(["c" + str(i) for i in range(len(ind.flatten()))])
		param_names = np.reshape(param_names, ind.shape)
		param_expression = ", ".join(param_names[ind].flatten()) + ","

		print("Param expression = " + param_expression)

		var_names = self.model.feature_names
		var_expression = ", ".join(var_names)

		print("Var expression = " + var_expression)

		symbolic_equations = self.model.symbolic_equations(param_names, self.__input_fmt)
		model_expression = ", ".join(symbolic_equations)

		print("Model expression = " + model_expression)

		wrapper = {}
		function = self.__create_model_func(param_expression, var_expression, model_expression, len(var_names))
		exec(function, wrapper)

		normalization_factor = np.ones(self.X.shape[1])
		if normalize:
			for i in range(self.X.shape[1]):
				normalization_factor[i] = np.max(self.X[:,i]) - np.min(self.X[:,i])

		X_norm = np.full(self.X.shape, normalization_factor)
		X = self.X/X_norm

		coef = self.model.coefficients()
		bounds = []
		for i, c in enumerate(coef[ind].flatten()):
			if c < 0.0:
				bounds.append([(1.0 + bounds_perc)*c, (1.0 - bounds_perc)*c])
			else:
				bounds.append([(1.0 - bounds_perc)*c, (1.0 + bounds_perc)*c])

		print("Bounds = " + str(bounds))

		np.random.seed(seed)
		p0 = np.zeros((number_of_restarts, coef[ind].shape[0]))
		for i in range(number_of_restarts):
			for j in range(coef[ind].shape[0]):
				p0[i,j] = np.random.uniform(low = bounds[j][0], high = bounds[j][1])

		# coef = np.zeros(self.model.coefficients().shape)
		# for i in range(coef.shape[0]):
		# 	popt, pcov = curve_fit(wrapper['func'](self.X0, normalization_factor, i), self.t, self.X[:, i]/normalization_factor[i], p0 = self.model.coefficients()[ind]) # sigma = X_std[:, i]
		# 	coef[ind] = popt

		all_coef = [None]*number_of_restarts
		coef = np.zeros(self.model.coefficients().shape)
		residuals = np.full(number_of_restarts, np.inf)
		for i in range(number_of_restarts):
			try:
				popt, pcov = curve_fit(wrapper['func'](self.X0, normalization_factor), self.t, X.flatten(), p0 = p0[i,:])
				coef[ind] = popt
				self.model.coefficients(coef)
				all_coef[i] = coef

				solution = self.model.simulate(self.X0, t = self.t)
				solution_norm = np.full(solution.shape, normalization_factor)
				solution /= solution_norm
				residuals[i] = np.sum((X.flatten() - solution.flatten())**2.0)
			except:
				continue

		self.model.coefficients(all_coef[np.argmin(residuals)])

	def __update_bounds_if_necessary(self, coefficients, bounds, bounds_perc, bounds_tol = 1.0e-3):
		new_bounds = []
		for i, c in enumerate(coefficients):
			lower_bound = bounds[i][0]
			upper_bound = bounds[i][1]
			if abs(c - lower_bound) < bounds_tol*abs(c):
				if c < 0.0:
					new_bounds.append([(1.0 + bounds_perc)*lower_bound, upper_bound])
				else:
					new_bounds.append([(1.0 - bounds_perc)*lower_bound, upper_bound])
			elif abs(c - upper_bound) < bounds_tol*abs(c):
				if c < 0.0:
					new_bounds.append([lower_bound, (1.0 - bounds_perc)*upper_bound])
				else:
					new_bounds.append([lower_bound, (1.0 + bounds_perc)*upper_bound])
			else:
				new_bounds.append(bounds[i])

		return new_bounds

	def __check_bounds(self, bounds, new_bounds, bounds_tol = 1.0e-3):
		bounds_array = np.array(bounds)
		new_bounds_array = np.array(new_bounds)
		error = np.abs(new_bounds_array - bounds_array)
		tol = bounds_tol*np.abs(bounds_array)

		# for i in range(error.shape[0]):
		# 	for j in range(error.shape[1]):
		# 		if error[i,j] > tol[i,j]:
		# 			return True

		# return False

		return np.any(error > tol)

	def differential_evolution(self, normalize = False, bounds_perc = 0.99, number_of_restarts = 20,
		strategy = 'best1bin', maxiter = 500, popsize_factor = 5, tol = 1.0e-10,
		mutation = (0.5, 1), recombination = 0.7, polish = True, disp = True,
		seed = 7):
		print("*** Using Differential Evolution method ***\n")

		ind = self.model.coefficients() != 0.0
		param_names = np.array(["c" + str(i) for i in range(len(ind.flatten()))])
		param_names = np.reshape(param_names, ind.shape)
		param_expression = ", ".join(param_names[ind].flatten()) + ","

		var_names = self.model.feature_names
		var_expression = ", ".join(var_names)

		symbolic_equations = self.model.symbolic_equations(param_names, self.__input_fmt)
		model_expression = ", ".join(symbolic_equations)

		wrapper = {}
		function = self.__create_ssr_func(param_expression, var_expression, model_expression, len(var_names))
		exec(function, wrapper)

		normalization_factor = np.ones(self.X.shape[1])
		if normalize:
			for i in range(self.X.shape[1]):
				normalization_factor[i] = np.max(self.X[:,i]) - np.min(self.X[:,i])

		coef = self.model.coefficients()
		bounds = []
		for i, c in enumerate(coef[ind].flatten()):
			if c < 0.0:
				bounds.append([(1.0 + bounds_perc)*c, (1.0 - bounds_perc)*c])
			else:
				bounds.append([(1.0 - bounds_perc)*c, (1.0 + bounds_perc)*c])

		# Opções para a otimização
		popsize = popsize_factor*len(coef[ind].flatten())
		options = {
			'strategy': strategy,			# Estratégia de mutação e recombinação
			'maxiter': maxiter,				# Número máximo de gerações/iterações para a evolução diferencial
			'popsize': popsize,				# Tamanho da população
			'tol': tol,						# Tolerância para a convergência da otimização
			'mutation': mutation,			# Valor de mutação para a evolução diferencial
			'recombination': recombination,	# Taxa de recombinação para a evolução diferencial
			'polish': polish,				# Se deve ser aplicado o método L-BFGS-B após a evolução diferencial para refinar o resultado
			'disp': disp,					# Se exibe mensagens de status durante a otimização
			'seed': seed					# Semente aleatória para reprodução dos resultados
		}

		for restart in range(1, number_of_restarts+1):
			print("Restart = " + str(restart))

			# Ajuste do modelo usando Evolução Diferencial
			if restart == 1:
				result = differential_evolution(wrapper['func'](self.X0, normalization_factor), bounds, args=(self.t, self.X), **options)
			else:
				result = differential_evolution(wrapper['func'](self.X0, normalization_factor), bounds, args=(self.t, self.X), x0=result.x, **options)

			new_bounds = self.__update_bounds_if_necessary(result.x, bounds, bounds_perc)
			updated_bounds = self.__check_bounds(bounds, new_bounds)
			if not updated_bounds:
				break
			
			bounds = new_bounds

		print("Success = " + str(result.success))
		print("Message = " + result.message)

		# Obter os parâmetros estimados
		coef = np.zeros(self.model.coefficients().shape)
		coef[ind] = result.x

		self.model.coefficients(coef)

	def bayesian_calibration(self, bounds_perc = 0.2, sd_bound_perc = 0.1, draws = 2500, seed = 7):
		print("*** Performing Bayesian calibration ***\n")

		coef = self.model.coefficients()
		ind = coef != 0.0
		param_names = np.array(["c" + str(i) for i in range(len(ind.flatten()))])
		param_names = np.reshape(param_names, ind.shape)
		param_expression = ", ".join(param_names[ind].flatten()) + ","

		var_names = self.model.feature_names
		var_expression = ", ".join(var_names)

		symbolic_equations = self.model.symbolic_equations(param_names, self.__input_fmt)
		model_expression = ", ".join(symbolic_equations)

		init_cond_names = [f + "0" for f in self.model.feature_names]
		init_cond_expression = ", ".join(init_cond_names) + ","
		
		itypes_names = ["tt.dscalar"]*(len(param_names[ind].flatten()) + len(init_cond_names))
		itypes_expression = ", ".join(itypes_names) + ","

		bounds = []
		for c in np.append(coef[ind].flatten(), self.X0):
			if c < 0.0:
				bounds.append([(1.0 + bounds_perc)*c, (1.0 - bounds_perc)*c])
			else:
				bounds.append([(1.0 - bounds_perc)*c, (1.0 + bounds_perc)*c])
		sd_upper_bound = sd_bound_perc*max(self.X.min(), self.X.max(), key = abs)

		wrapper = {}
		code = self.__create_model(param_expression, var_expression, model_expression, len(var_names))
		exec(code, wrapper)

		code = self.__create_model_wrapper(itypes_expression, param_expression, init_cond_expression, len(var_names))
		exec(code, wrapper)

		print("-- Running Monte Carlo simulations:\n")
		start_time = time.time()
		code = self.__create_smc_code(np.append(param_names[ind].flatten(), init_cond_names), bounds, sd_upper_bound, len(self.model.feature_names), param_expression, init_cond_expression, draws, seed)
		exec(code)
		duration = time.time() - start_time
		print(f"\n-- Monte Carlo simulations done in {duration / 60:.3f} minutes\n")

		self.calibration_param_names = np.append(param_names[ind].flatten(), np.append(init_cond_names, ["std_deviation"]))

	def traceplot(self, plot_step = 1):
		progressbar = tqdm(self.calibration_param_names)
		for param_name in progressbar:
			progressbar.set_description("Arviz post-processing")
			pm.traceplot(self.trace_calibration[::plot_step], var_names = (f"{param_name}"))
			plt.savefig(os.path.join("output", f"cal_traceplot_model{str(self.model_id+1)}_ic{str(self.init_cond_id)}_{param_name}.png"), bbox_inches = 'tight')
			plt.close()

	def plot_posterior(self, plot_step = 1):
		progressbar = tqdm(self.calibration_param_names)
		for param_name in progressbar:
			pm.plot_posterior(
				self.trace_calibration[::plot_step],
				var_names = (f"{param_name}"),
				kind = "hist",
				round_to = 5,
				point_estimate = "mode",
				figsize = (10, 10)
			)
			plt.savefig(os.path.join("output", f"cal_posterior_model{str(self.model_id+1)}_ic{str(self.init_cond_id)}_{param_name}.png"), bbox_inches = 'tight')
			plt.close()

	def plot_pair(self):
		az.plot_pair(
			self.trace_calibration,
			var_names = self.calibration_param_names[:-1],
			kind = "hexbin",
			fill_last = False,
			figsize = (10, 10)
		)
		plt.savefig(os.path.join("output", f"cal_marginals_model{str(self.model_id+1)}_ic{str(self.init_cond_id)}.png"), bbox_inches = 'tight')
		plt.close()

	def summary(self):
		df_stats_summary = az.summary(
			data = self.trace_calibration,
			var_names = self.calibration_param_names,
			kind = "stats",
			round_to = 15
		)
		calibration_param_mpv = self.__calculate_rv_posterior_mpv(pm_trace = self.trace_calibration, variable_names = self.calibration_param_names)
		df_stats_summary = self.__add_mpv_to_summary(df_stats_summary, calibration_param_mpv)
		df_stats_summary.to_csv(os.path.join("output", f"cal_stats_summary_modelo{str(self.model_id+1)}_ci{str(self.init_cond_id+1)}.csv"))

		coef = self.model.coefficients()
		ind = coef != 0.0
		k = np.count_nonzero(ind)
		coef[ind] = df_stats_summary.iloc[:k]['mpv']
		self.model.coefficients(coef)

		return df_stats_summary.iloc[k:(k + len(self.model.feature_names))]['mpv'].to_numpy()

	def get_simulation(self, percentile_cut = 2.5):
		simulation = np.percentile(self.trace_calibration["model"], 50, axis = 0)
		simulation_min = np.percentile(self.trace_calibration["model"], percentile_cut, axis = 0)
		simulation_max = np.percentile(self.trace_calibration["model"], 100 - percentile_cut, axis = 0)

		return (simulation, simulation_min, simulation_max)