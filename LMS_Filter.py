import numpy as np
from scipy.io.wavfile import read


def gradient_descent(y, y_hat, x, lms_filter, step):
    return lms_filter + 2*step*(y-y_hat)*x


def lms_filter(sound, p):
    lms_filter = np.zeros((sound.shape[0], p))
    estimate = np.zeros(sound.shape)
    lms_filter[0] = np.random.randn(p)
    for time in range(p, sound.shape[0]):
        estimate[time] = lms_filter[time-p] @ sound[time-p:time]
        lms_filter[time] = gradient_descent(sound[time], estimate[time], sound[time-p:time], lms_filter[time-1], 0.01)
    return estimate
    

def find_best_parameter_lms(sound, original_sound, nombre_params_a_tester ):
    reconstructions = [sound - lms_filter(sound, i) for i in range(1,nombre_params_a_tester)]
    scores = [np.linalg.norm(original_sound - reconstruction) for reconstruction in reconstructions]
    return np.argmin(np.array(scores))

def get_sound_and_normalize(file_name):
    rate, sound = read(file_name)
    sound = sound.astype(np.float64)
    sound -= sound.mean()
    sound /= sound.std()
    return rate, sound