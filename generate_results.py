import numpy as np
import stream_utils as streams
import early_stopping
import lstsq
import mlp

def lstsq_model_selection():
	print '-- Least Squares Regression ----------- :'
	splits = 10
	seed = 123456
	model_params = np.arange(0.0,21*5e-2,5e-2)
	model_testing_errors = np.empty(shape=(model_params.size, 1 + splits))
	model_testing_classerrors = np.empty(shape=(model_params.size, 1 + splits))
	for i,v in enumerate(model_params):
		print ' - Generating for param (v)             :', v
		classifier = lstsq.LeastSquares(v)
		errors = classerrors = []
		for s,(training_stream,validation_stream) in enumerate(streams.cross_validation_5class(splits=splits, seed=seed)):
			print '   -> cross validation                  :', s+1, '/', splits
			x_left, x_right, t = training_stream.all()
			classifier.train(x_left, x_right, t)
			error, classerror = classifier.normalized_error(validation_stream)
			errors, classerrors = errors + [error], classerrors + [classerror]
		print '   average error                        :', float(sum(errors)) / float(len(errors))
		print '   average classification errors        :', float(sum(classerrors)) / float(len(classerrors))
		model_testing_errors[i] = [v] + errors
		model_testing_classerrors[i] = [v] + classerrors
	np.savetxt('results/lstsq_errors.txt', model_testing_errors)
	np.savetxt('results/lstsq_classerrors.txt', model_testing_classerrors)

def cartesian(x, y):
	return numpy.dstack(numpy.meshgrid(x, y)).reshape(-1, 2)

nus = np.array([
	('0.01', 0.01), ('0.02', 0.02), ('0.05', 0.05), ('1/x', lambda x: 1.0/float(x)), ('1/10x', lambda x: 0.1/float(x)),
	('1/100x', lambda x: 0.01/float(x)), ('1/sqrt(x)', lambda x: 1.0/pow(float(x),0.5)), ('1/10sqrt(x)', lambda x: 0.1/pow(float(x),0.5)),
	('1/100sqrt(x)', lambda x: 0.01/pow(float(x),0.5)), ('1/cbrt(x)', lambda x: 1.0/pow(float(x),0.33)),
	('1/10cbrt(x)', lambda x: 0.1/pow(float(x),0.33)), ('1/100cbrt(x)', lambda x: 0.01/pow(float(x),0.33))
])
mus = np.arange(0.05, 5*0.05, 0.05)
counts = np.array([1, 2, 5, 10, 20, 50])
#gradient_model_params = cartesian(nus, mus, counts)

def logistic_descent_model_selection():
	seed = 123541
	print '-- Logistic Regression ---------------- :'
	model_testing = np.empty(shape=(gradient_model_params.size, 7))
	for i,((nu_repr,nu),mu,count) in enumerate(gradient_model_params):
		print ' - Generating for params (nu,mu,count)  :', nu_repr, mu, count
		classifier = logistic.LogisticLoss(nu,mu)
		training, validation = streams.validation_5class(seed=seed)
		trained, chain, seconds = early_stopping.run(training, validation, classifier, count=count)
		error, classerror = trained.normalized_error(validation)
		print '   chain length                         :', len(chain)
		print '   seconds                              :', seconds
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [nu_repr, mu, count, len(chain), seconds, error, classerror]
	np.savetxt('results/logistic_descent.txt', model_testing)

def mlp_binary_descent_model_selection():
	seed = 2819732
	print '-- Binary MLP Gradient Descent -------- :'
	model_testing = np.empty(shape=(gradient_model_params.size, 7))
	for i,((nu_repr,nu),mu,count) in enumerate(gradient_model_params):
		print ' - Generating for params (nu,mu,count)  :', nu_repr, mu, count
		classifier = mlp.MLP(10, 10, nu=nu, mu=mu, k=2)
		training, validation = streams.validation_binary(seed=seed)
		trained, chain, seconds = early_stopping.run(training, validation, classifier, count=count)
		error, classerror = trained.normalized_error(validation)
		print '   chain length                         :', len(chain)
		print '   seconds                              :', seconds
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [nu_repr, mu, count, len(chain), seconds, error, classerror]
	np.savetxt('results/mlp_binary_descent.txt', model_testing)

def mlp_5class_descent_model_selection():
	seed = 12398123
	print '-- 5Class MLP Gradient Descent -------- :'
	model_testing = np.empty(shape=(gradient_model_params.size, 7))
	for i,((nu_repr,nu),mu,count) in enumerate(gradient_model_params):
		print ' - Generating for params (nu,mu,count)  :', nu_repr, mu, count
		classifier = mlp.MLP(10, 10, nu=nu, mu=mu, k=5)
		training, validation = streams.validation_5class(seed=seed)
		trained, chain, seconds = early_stopping.run(training, validation, classifier, count=count)
		error, classerror = trained.normalized_error(validation)
		print '   chain length                         :', len(chain)
		print '   seconds                              :', seconds
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [nu_repr, mu, count, len(chain), seconds, error, classerror]
	np.savetxt('results/mlp_5class_descent.txt', model_testing)

def mlp_binary_model_selection():
	seed = 293482934
	print '-- Binary MLP ------------------------- :'
	H1s = H2s = np.arange(10,9*10,10)
	model_params = cartesian(H1s, H2s)
	model_testing = np.empty(shape=(model_params.size, 4))
	for i,(H1,H2) in enumerate(model_params):
		print ' - Generating for params (H1,H2)        :', H1, H2
		classifier = mlp.MLP(H1, H2, k=2)
		training, validation = streams.validation_binary(seed=seed)
		trained, _, _ = early_stopping.run(training, validation, classifier, count=10)
		error, classerror = trained.normalized_error(validation)
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [H1, H2, error, classerror]
	np.savetxt('results/mlp_binary.txt', model_testing)

def mlp_5class_model_selection():
	seed = 23427310
	print '-- 5Class MLP ------------------------- :'
	H1s, H2s = np.arange(10,9*10,10)
	model_params = cartesian(H1s, H2s)
	model_testing = np.empty(shape=(model_params.size, 4))
	for i,(H1,H2) in enumerate(model_params):
		print ' - Generating for params (H1,H2)        :', H1, H2
		classifier = mlp.MLP(H1, H2, k=5)
		training, validation = streams.validation_5class(seed=seed)
		trained, _, _ = early_stopping.run(training, validation, classifier, count=10)
		error, classerror = trained.normalized_error(validation)
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [H1, H2, error, classerror]
	np.savetxt('result/mlp_5class.txt', model_testing)

lstsq_model_selection()
