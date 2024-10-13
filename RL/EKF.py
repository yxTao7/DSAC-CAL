import numpy as np
from ecm_parameter import get_parameters, dynamic_state, OCV_fromSOCtemp
import scipy.io
from scipy.interpolate import interp1d


mat_file = scipy.io.loadmat('A123model.mat')
model_struct = mat_file['model']


def dOCVfromSOC(SOC, OCV, z):
    dZ = SOC[1] - SOC[0]
    dUdZ = np.diff(OCV) / dZ
    dOCV = (np.concatenate(([dUdZ[0]], dUdZ)) + np.concatenate((dUdZ, [dUdZ[-1]]))) / 2
    interp_func = interp1d(SOC, dOCV, kind='linear', fill_value="extrapolate")
    dOCVz = interp_func(z)
    return dOCVz


def linMat(xhat, model, ik, deltaT):

    xhat[0] = min(1.0, max(-0.05, xhat[0]))
    zk = xhat[0]
    irk = xhat[1]
    hk = xhat[2]
    Tc = xhat[3]
    Ts = xhat[4]

    temp = Tc

    R = get_parameters('RParam', temp)
    RC = get_parameters('RCParam', temp)
    eta = get_parameters('etaParam', temp)
    Q = get_parameters('QParam', temp)
    M = get_parameters('MParam', temp)
    M0 = get_parameters('M0Param', temp)
    R0 = get_parameters('R0Param', temp)
    Rc = model_struct['Rc'][0, 0][0]
    Ru = model_struct['Ru'][0, 0][0]
    Cc = model_struct['Cc'][0, 0][0]
    Cs = model_struct['Cs'][0, 0][0]

    Am = np.zeros((5, 5))

    zkInd = 0
    irkInd = 1
    hkInd = 2
    TcInd = 3
    TsInd = 4

    n = eta

    A_RC = np.exp(-deltaT / RC)
    A_HK = np.exp(-abs(ik * n * deltaT / (3600 * Q)))

    Am[zkInd, zkInd] = 1
    Am[irkInd, irkInd] = A_RC
    Am[hkInd, hkInd] = A_HK
    Am[TcInd, irkInd] = (deltaT / Cc) * R * ik
    Am[TcInd, hkInd] = -M * ik * deltaT / Cc
    Am[TcInd, TcInd] = 1 - deltaT / Rc / Cc
    Am[TcInd, TsInd] = deltaT / Rc / Cc
    Am[TsInd, TcInd] = deltaT / Rc / Cc
    Am[TsInd, TsInd] = 1 - deltaT / Rc / Cs - deltaT / Ru / Cs

    Bm = np.zeros(5)
    Bd1 = np.zeros(5)
    Bd2 = np.zeros(5)

    Bm[zkInd] = -deltaT * n / (3600 * Q)
    Bm[irkInd] = 1 - A_RC
    Bm[TcInd] = (deltaT / Cc) * (R * irk + 2 * R0 * ik - M * hk)

    Bd1[hkInd] = -abs(deltaT * n / (3600 * Q)) * A_HK * (1 + np.sign(ik) * hk)
    Bd2[TsInd] = deltaT / Ru / Cs

    Bhat = np.column_stack((Bm + Bd1, Bd2))

    Chat = np.zeros((2, 5))
    dOCVdSOC = dOCVfromSOC(model['SOC'][0, 0].flatten(), model['OCV0'][0, 0].flatten(), zk)
    Chat[0, :] = [dOCVdSOC, -R, M, 0, 0]
    Chat[1, :] = [0, 0, 0, deltaT / Rc / Cc, 1 - deltaT / Rc / Cs - deltaT / Ru / Cs]

    C_v = [dOCVdSOC, -R, M, 0, 0]
    C_z = [1, 0, 0, 0, 0]
    C_Tc = [0, 0, 0, 1, 0]

    Dhat = 1
    D_v = -R0
    D_Tc = 0
    D_z = 0

    linMatrices = {
        'A': Am,
        'B': Bm + Bd1,
        'Bd1': Bd1,
        'Bd2': Bd2,
        'Bhat': Bhat,
        'Chat': Chat,
        'Dhat': Dhat,
        'C_v': C_v,
        'C_z': C_z,
        'C_Tc': C_Tc,
        'D_v': D_v,
        'D_z': D_z,
        'D_Tc': D_Tc,
        'dOCVdSOC': dOCVdSOC
    }

    return linMatrices


def initEKF(x):

    ekfData = {
        'zkInd': 1,
        'irInd': 2,
        'hkInd': 3,
        'TcInd': 4,
        'TsInd': 5,
        'thetaInd': 6,
        'xhat': x,
        'SigmaX': np.diag([1e-3, 1e-8, 1e-7, 1e-6, 1e-4]),
        'SigmaV': np.diag([1e-3, 1e-4]),
        'SigmaW': np.diag([2e-7, 1.6576e-5]),
        'Qbump': 5,
        'SOC': x[0],
        'prior_SOC': x[0],  # Past SOC guess
        'uk_1': 0,  # Current applied before first time step
        'signIk': 0,
        'deltaT': 1
    }

    OCV = OCV_fromSOCtemp(x[0], x[3], model_struct)
    ekfData['OCV_1'] = OCV
    ekfData['v_1'] = OCV

    linMatrices = linMat(x, model_struct, ekfData['uk_1'], ekfData['deltaT'])

    return linMatrices, ekfData


def iterEKF(voltage, Tsk, uk, Tfk, xhat_plus, ekfData):

    uk_1 = ekfData['uk_1']
    deltaT = ekfData['deltaT']

    linMatrices = linMat(xhat_plus, model_struct, uk_1, deltaT)
    Ahat = linMatrices['A']
    Bhat = linMatrices['Bhat']
    Chat = linMatrices['Chat']
    Dhat = linMatrices['Dhat']

    SigmaX = ekfData['SigmaX']
    SigmaW = ekfData['SigmaW']
    SigmaV = ekfData['SigmaV']

    # Step 1a: State estimate time update
    xhat_minus, vhat = est_state(xhat_plus, uk, uk_1, Tfk, deltaT)
    xhat_minus[3] = xhat_minus[3] + 273.15
    xhat_minus[4] = xhat_minus[4] + 273.15

    # Step 1b: Error covariance time update
    SigmaX = Ahat @ SigmaX @ Ahat.T + Bhat @ SigmaW @ Bhat.T
    SigmaX = makePositive(SigmaX)

    # Step 1c: Output prediction
    # Computed in Step 1a

    # Step 2a: Kalman gain calculation
    SigmaY = Chat @ SigmaX @ Chat.T + Dhat * SigmaV * Dhat
    L = SigmaX @ Chat.T @ np.linalg.inv(SigmaY)

    # Step 2b: State estimate measurement update
    residual = np.array([voltage - vhat, Tsk - xhat_minus[4]]).reshape(2, 1)
    xhat_plus = xhat_minus + (L @ residual).flatten()

    # Step 2c: Error covariance measurement update
    SigmaX = SigmaX - L @ SigmaY @ L.T
    SigmaX = makePositive(SigmaX)

    # Update ekf related variables
    ekfData['uk_1'] = uk
    ekfData['Tf_1'] = Tfk
    ekfData['prior_SOC'] = xhat_plus[0]
    ekfData['SigmaX'] = SigmaX

    xhat_plus = np.array(xhat_plus)
    xhat_plus[3] = xhat_plus[3] - 273.15
    xhat_plus[4] = xhat_plus[4] - 273.15

    return linMatrices, xhat_plus, ekfData, voltage


def makePositive(x):
    # Make sure x is neither NaN nor Inf
    if np.all(np.isfinite(x)):
        _, S, Vt = np.linalg.svd(x)
        HH = Vt.T @ np.diag(S) @ Vt
        y = (x + x.T + HH + HH.T) / 4
    else:
        y = x
    return y


def est_state(xhat, ik, ik_old, Tfk, deltaT):

    temp = xhat[3]
    R = get_parameters('RParam', temp)
    RC = get_parameters('RCParam', temp)
    eta = get_parameters('etaParam', temp)
    Q = get_parameters('QParam', temp)
    M = get_parameters('MParam', temp)
    M0 = get_parameters('M0Param', temp)
    R0 = get_parameters('R0Param', temp)
    Rc = model_struct['Rc'][0, 0][0]
    Ru = model_struct['Ru'][0, 0][0]
    Cc = model_struct['Cc'][0, 0][0]
    Cs = model_struct['Cs'][0, 0][0]

    zk = xhat[0]
    irk = xhat[1]
    hk = xhat[2]

    if type(zk) != type(irk):
        zk = np.array([xhat[0]])

    xhat[3] = xhat[3] + 273.15
    xhat[4] = xhat[4] + 273.15

    # Compute output equation
    OCV = OCV_fromSOCtemp(zk, temp, model_struct)
    vk = OCV - irk * R - ik_old * R0 + M * hk - M0 * np.sign(ik_old)

    A_RC = np.exp(-deltaT / RC)
    A_HK = np.exp(-abs(ik_old * eta * deltaT / (3600 * Q)))

    A = np.array([
        [1, 0, 0, 0, 0],
        [0, A_RC, 0, 0, 0],
        [0, 0, A_HK, 0, 0],
        [0, 0, 0, (1 - deltaT / Rc / Cc), (deltaT / Rc / Cc)],
        [0, 0, 0, (deltaT / Rc / Cs), (1 - deltaT / Rc / Cs - deltaT / Ru / Cs)]
    ],dtype=object)

    B = np.array([
        [-deltaT * eta / (3600 * Q), 0, 0],
        [(1 - A_RC), 0, 0],
        [0, (A_HK - 1), 0],
        [(OCV - vk) * deltaT / Cc, 0, 0],
        [0, 0, deltaT / Ru / Cs]
    ],dtype=object)

    xhat = A @ xhat + B @ np.array([ik_old, np.sign(ik_old), Tfk])

    zkhat = xhat[0]
    irhat = xhat[1]
    hk = xhat[2]

    OCV = OCV_fromSOCtemp(zkhat, xhat[3] - 273.15, model_struct)
    vhat = OCV + M * hk - R * irhat - R0 * ik - M0 * np.sign(ik)
    xhat[3] = xhat[3] - 273.15
    xhat[4] = xhat[4] - 273.15

    return xhat, vhat