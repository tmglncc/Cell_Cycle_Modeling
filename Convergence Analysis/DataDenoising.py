import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylops
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RBF, WhiteKernel, RationalQuadratic, DotProduct

class DataDenoising:
	def __init__(self, X = None, t = None, feature_names = None):
		self.X = X
		self.t = t
		self.feature_names = feature_names

		if self.X is not None and self.t is not None:
			data = np.zeros((X.shape[0], X.shape[1]+1))
			data[:,0] = self.t
			data[:,1:] = self.X

			columns = ["t"] + feature_names

			self.dataset = pd.DataFrame(
				data = data,
				columns = columns
			)
	
	def simple_moving_average(self, window = 3):
		dataset_copy = self.dataset.iloc[:,1:].copy()
		for column in dataset_copy:
			dataset_copy[column] = dataset_copy[column].rolling(window, min_periods = 1).mean()

		return dataset_copy.to_numpy()

	def exponential_moving_average(self, alpha = 0.1, adjust = False):
		dataset_copy = self.dataset.iloc[:,1:].copy()
		for column in dataset_copy:
			dataset_copy[column] = dataset_copy[column].ewm(alpha = alpha, adjust = adjust).mean()

		return dataset_copy.to_numpy()

	def l2_regularization(self, lambda_ = 1.0e2):
		Iop = pylops.Identity(self.t.shape[0])
		D2op = pylops.SecondDerivative(self.t.shape[0], edge = True)

		X_l2r = np.zeros(self.X.shape)
		for j in range(X_l2r.shape[1]):
			Y = Iop*self.X[:,j]
			X_l2r[:,j] = pylops.optimization.leastsquares.RegularizedInversion(Iop, [D2op], Y, 
				epsRs = [np.sqrt(lambda_/2.0)], 
				**dict(iter_lim = 30)
			)

		return X_l2r

	def total_variation_regularization(self, mu = 0.01, lambda_ = 0.3, niter_out = 50, niter_in = 3):
		Iop = pylops.Identity(self.t.shape[0])
		Dop = pylops.FirstDerivative(self.t.shape[0], edge = True, kind = 'backward')

		X_tvr = np.zeros(self.X.shape)
		for j in range(X_tvr.shape[1]):
			Y = Iop*self.X[:,j]
			X_tvr[:,j], niter = pylops.optimization.sparsity.SplitBregman(Iop, [Dop], Y, 
				niter_out, niter_in, mu = mu, epsRL1s = [lambda_], 
				tol = 1.0e-4, tau = 1.0, 
				**dict(iter_lim = 30, damp = 1.0e-10)
			)

		return X_tvr

	def gaussian_process_regression(self, kernel = RBF(), n_restarts_optimizer = 10, alpha = 1.0e-10, t_pred = None):
		if t_pred is None:
			t_pred = self.t

		X_gpr_mean = np.zeros((t_pred.shape[0], self.X.shape[1]))
		X_gpr_min = np.zeros((t_pred.shape[0], self.X.shape[1]))
		X_gpr_max = np.zeros((t_pred.shape[0], self.X.shape[1]))
		for j in range(X_gpr_mean.shape[1]):
			model = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = n_restarts_optimizer, alpha = alpha, normalize_y = np.max(self.X[:,j]))
			model.fit(self.t.reshape(-1, 1), self.X[:,j])
			target_pred = model.predict(t_pred.reshape(-1, 1), return_std = True)
			# target_pred[0]: média, target_pred[1]: desvio-padrão

			X_gpr_mean[:,j] = target_pred[0]
			error = 1.96*np.max(self.X[:,j])*target_pred[1]
			X_gpr_min[:,j] = target_pred[0] - error
			X_gpr_max[:,j] = target_pred[0] + error

		return (X_gpr_mean, X_gpr_min, X_gpr_max)

	def plot_derivative(self, X_dot = None, t = None, init_cond_id = None, X0 = None):
		if X_dot is not None and t is not None:
			markers = ["o", "^", "s", "p", "P", "*", "X", "d"]

			fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
			for i, feature_name in enumerate(self.feature_names):
				ax.plot(t, X_dot[:,i], "k" + markers[i], label = r"$" + feature_name + "'(t)$", alpha = 0.5, markersize = 3)
			for i, feature_name in enumerate(self.feature_names):
				ax.plot(t, X_dot[:,i], label = r"$" + feature_name + "'(t)$", alpha = 1.0, linewidth = 1)
			ax.set(xlabel = r"Time $t$", ylabel = r"$X'(t)$",
				# title = "Derivative - Initial condition = " + str(X0)
			)
			ax.legend()
			# fig.show()
			plt.savefig(os.path.join("output", "deriv_ic" + str(init_cond_id) + ".png"), bbox_inches = 'tight')
			plt.close()

	def plot_sma(self, windows = None):
		if windows is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for window in windows:
					X_sma = self.simple_moving_average(window)
					ax.plot(self.t, X_sma[:,i], label = "SMA(" + str(window) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"Simple Moving Average - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "SMA_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_ema(self, alphas = None, adjusts = None):
		if alphas is not None and adjusts is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for alpha in alphas:
					for adjust in adjusts:
						X_ema = self.exponential_moving_average(alpha, adjust)
						ax.plot(self.t, X_ema[:,i], label = "EMA(" + str(alpha) + ", " + str(adjust) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"Exponential Moving Average - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "EMA_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_l2r(self, lambdas = None):
		if lambdas is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for lambda_ in lambdas:
					X_l2r = self.l2_regularization(lambda_ = lambda_)
					ax.plot(self.t, X_l2r[:,i], label = "L2R(" + str(lambda_) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"L2 Regularization - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "L2R_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_tvr(self, mus = None, lambdas = None):
		if mus is not None and lambdas is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for mu in mus:
					for lambda_ in lambdas:
						X_tvr = self.total_variation_regularization(mu = mu, lambda_ = lambda_)
						ax.plot(self.t, X_tvr[:,i], label = "TVR(" + str(mu) + ", " + str(lambda_) + ")", alpha = 1.0, linewidth = 1)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"Total Variation Regularization - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig(os.path.join("output", "TVR_" + self.feature_names[i] + ".png"), bbox_inches = 'tight')
				plt.close()

	def plot_gpr(self, kernels = None, n_restarts_optimizers = None, alphas = None, kernel_strings = None, set_id = ""):
		if kernels is not None and n_restarts_optimizers is not None and alphas is not None:
			for i, feature_name in enumerate(self.feature_names):
				fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
				ax.plot(self.t, self.X[:,i], "ko", label = r"$" + self.feature_names[i] + "(t)$", alpha = 0.5, markersize = 3)
				for j, kernel in enumerate(kernels):
					for n_restarts_optimizer in n_restarts_optimizers:
						for alpha in alphas:
							X_gpr_mean, X_gpr_min, X_gpr_max = self.gaussian_process_regression(kernel = kernel, n_restarts_optimizer = n_restarts_optimizer, alpha = alpha)
							ax.plot(self.t, X_gpr_mean[:,i], label = "GPR(" + kernel_strings[j] + ", " + str(n_restarts_optimizer) + ", " + str(alpha) + ")", alpha = 1.0, linewidth = 1)
							ax.fill_between(self.t, X_gpr_min[:,i], X_gpr_max[:,i], alpha = 0.4)
				ax.set(xlabel = r"Time $t$", ylabel = r"$" + self.feature_names[i] + "(t)$",
					# title = r"Gaussian Process Regression - $" + self.feature_names[i] + "(t)$"
				)
				ax.legend()
				# fig.show()
				plt.savefig("GPR" + str(set_id) + "_" + self.feature_names[i] + ".png", bbox_inches = 'tight')
				plt.close()
