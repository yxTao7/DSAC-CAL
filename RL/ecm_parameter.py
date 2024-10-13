import numpy as np
import math
import scipy.io
from scipy.interpolate import interp1d

mat_file = scipy.io.loadmat('A123model.mat')
model_struct = mat_file['model']


def entropy_variation(soc):
    # 读取 entropy.mat 文件
    data = scipy.io.loadmat('entropy.mat')

    SOCs = data['SOCs'].flatten()/100
    dOCVdT = data['dOCVdT'].flatten()

    # 创建插值函数
    interp_func = interp1d(SOCs, dOCVdT, kind='linear', fill_value="extrapolate")

    # 进行插值
    dVdT = interp_func(soc)

    return dVdT


def OCV_fromSOCtemp(soc, temp, model):

    soccol = soc.flatten()

    # Force to be column vectors
    SOC = model['SOC'][0, 0].flatten()
    OCV0 = model['OCV0'][0, 0].flatten()
    OCVrel = model['OCVrel'][0, 0].flatten()

    # Handle temperature input
    if np.isscalar(temp):
        tempcol = temp * np.ones_like(soccol)
    else:
        tempcol = temp.flatten()

    # Check if temperature and SOC vectors are of the same length
    if len(tempcol) != len(soccol):
        raise ValueError('Temperature must be a scalar, or a vector the same length as SOC')

    diffSOC = SOC[1] - SOC[0]
    ocv = np.zeros_like(soccol)

    I1 = np.where(soccol <= SOC[0])[0]  # Low SOC indices
    I2 = np.where(soccol >= SOC[-1])[0]  # High SOC indices
    I3 = np.where((soccol > SOC[0]) & (soccol < SOC[-1]))[0]  # Normal SOC range indices
    I6 = np.isnan(soccol)  # NaN SOC indices

    # Extrapolate for SOCs less than the minimum SOC
    if I1.size > 0:
        dv = (OCV0[1] + tempcol * OCVrel[1]) - (OCV0[0] + tempcol * OCVrel[0])
        ocv[I1] = (soccol[I1] - SOC[0]) * dv[I1] / diffSOC + OCV0[0] + tempcol[I1] * OCVrel[0]

    # Extrapolate for SOCs greater than the maximum SOC
    if I2.size > 0:
        dv = (OCV0[-1] + tempcol * OCVrel[-1]) - (OCV0[-2] + tempcol * OCVrel[-2])
        ocv[I2] = (soccol[I2] - SOC[-1]) * dv[I2] / diffSOC + OCV0[-1] + tempcol[I2] * OCVrel[-1]

    # Interpolate for SOCs within the normal range
    if I3.size > 0:
        I4 = (soccol[I3] - SOC[0]) / diffSOC
        I5 = np.floor(I4).astype(int)
        I45 = I4 - I5
        omI45 = 1 - I45
        ocv[I3] = OCV0[I5] * omI45 + OCV0[I5 + 1] * I45
        ocv[I3] += tempcol[I3] * (OCVrel[I5] * omI45 + OCVrel[I5 + 1] * I45)

    # Replace NaN SOCs with zero voltage
    ocv[I6] = 0

    return ocv.reshape(soc.shape)


def SOC_fromOCVtemp(ocv, temp, model):
    ocvcol = np.array(ocv).flatten()  # 将ocv转换为列向量
    OCV = np.array(model['OCV'])[0, 0].flatten()  # 将OCV转换为列向量
    SOC0 = np.array(model['SOC0'])[0, 0].flatten()  # 将SOC0转换为列向量
    SOCrel = np.array(model['SOCrel'])[0, 0].flatten()  # 将SOCrel转换为列向量

    if np.isscalar(temp):  # 如果temp是标量，复制成与ocvcol相同大小
        tempcol = temp * np.ones(ocvcol.shape)
    else:  # 如果temp是数组，将其转换为列向量
        tempcol = np.array(temp).flatten()
        if tempcol.shape != ocvcol.shape:
            raise ValueError(
                'Function inputs "ocv" and "temp" must either have same number of elements, or "temp" must be a scalar')

    diffOCV = OCV[1] - OCV[0]  # 假设OCV点之间均匀间隔

    soc = np.zeros(ocvcol.shape)  # 初始化输出为零
    I1 = np.where(ocvcol <= OCV[0])[0]  # ocv低于模型存储数据的索引
    I2 = np.where(ocvcol >= OCV[-1])[0]  # ocv高于模型存储数据的索引
    I3 = np.where((ocvcol > OCV[0]) & (ocvcol < OCV[-1]))[0]  # 其余的索引
    I6 = np.where(np.isnan(ocvcol))[0]  # ocv为NaN的位置

    # 对于低于最低电压的ocv，进行外插
    if I1.size > 0:
        dz = (SOC0[1] + tempcol * SOCrel[1]) - (SOC0[0] + tempcol * SOCrel[0])
        soc[I1] = (ocvcol[I1] - OCV[0]) * dz[I1] / diffOCV + SOC0[0] + tempcol[I1] * SOCrel[0]

    # 对于高于最高电压的ocv，进行外插
    if I2.size > 0:
        dz = (SOC0[-1] + tempcol * SOCrel[-1]) - (SOC0[-2] + tempcol * SOCrel[-2])
        soc[I2] = (ocvcol[I2] - OCV[-1]) * dz[I2] / diffOCV + SOC0[-1] + tempcol[I2] * SOCrel[-1]

    # 对于正常范围内的ocv，手动插值
    I4 = (ocvcol[I3] - OCV[0]) / diffOCV  # 线性插值
    I5 = np.floor(I4).astype(int)
    I45 = I4 - I5
    omI45 = 1 - I45
    soc[I3] = SOC0[I5 + 0] * omI45 + SOC0[I5 + 1] * I45
    soc[I3] += tempcol[I3] * (SOCrel[I5 + 0] * omI45 + SOCrel[I5 + 1] * I45)
    soc[I6] = 0  # 将NaN OCV替换为零SOC

    soc = soc.reshape(ocv.shape)  # 输出与输入形状相同

    return soc


def get_parameters(paramName, temp):

    temp = min(temp, max(model_struct['temps'][0,0].flatten()))
    temp = max(temp, min(model_struct['temps'][0,0].flatten()))

    # 查找满足条件的索引
    ind = np.where(model_struct['temps'][0,0].flatten() == temp)[0]

    # 检查索引是否为空
    if ind.size > 0:
        # 根据参数名称获取参数值
        if paramName.upper() == 'QPARAM':
            theParam = (model_struct['QParam'][0, 0].flatten())[ind]
        elif paramName.upper() == 'RCPARAM':
            theParam = model_struct['RCParam'][0,0][ind,0]
        elif paramName.upper() == 'RPARAM':
            theParam = model_struct['RParam'][0,0][ind,0]
        elif paramName.upper() == 'R0PARAM':
            theParam = (model_struct['R0Param'][0,0].flatten())[ind]
        elif paramName.upper() == 'MPARAM':
            theParam = (model_struct['MParam'][0,0].flatten())[ind]
        elif paramName.upper() == 'M0PARAM':
            theParam = (model_struct['M0Param'][0,0].flatten())[ind]
        elif paramName.upper() == 'GPARAM':
            theParam = (model_struct['GParam'][0,0].flatten())[ind]
        elif paramName.upper() == 'ETAPARAM':
            theParam = (model_struct['etaParam'][0,0].flatten())[ind]
        else:
            raise ValueError('Bad argument to "paramName"')
    else:
        # 插值操作
        interp = interp1d(model_struct['temps'][0, 0].flatten(), model_struct[paramName][0, 0].flatten(), kind='cubic', fill_value="extrapolate")
        theParam = interp(temp)

    return theParam


def dynamic_state(x, u, Tf):

    dt = 1
    temp = x[3]
    R = get_parameters('RParam', temp)
    RC = get_parameters('RCParam', temp)
    eta = get_parameters('etaParam', temp)
    Q = get_parameters('QParam', temp)
    M = get_parameters('MParam', temp)
    M0 = get_parameters('M0Param', temp)
    R0 = get_parameters('R0Param', temp)
    Rc = model_struct['Rc'][0,0][0]
    Ru = model_struct['Ru'][0,0][0]
    Cc = model_struct['Cc'][0,0][0]
    Cs = model_struct['Cs'][0,0][0]

    u = eta * u    # u<0时
    zk, ir, hk = x[0], x[1], x[2]
    if type(zk) != type(ir):
        zk = np.array([x[0]])
    # 计算dOCVdT和OCV
    dOCVdT = entropy_variation(zk)
    OCV = OCV_fromSOCtemp(zk, temp, model_struct)
    v = OCV + M * hk - M0 * np.sign(u) - R * ir - R0 * u

    # 计算状态转移矩阵A和输入矩阵B
    A_RC = np.exp(-dt / RC)
    A_HK = np.exp(-abs(u * eta * dt / (3600 * Q)))

    A = np.array([
        [1, 0, 0, 0, 0],
        [0, A_RC, 0, 0, 0],
        [0, 0, A_HK, 0, 0],
        [0, 0, 0, 1 - dt / Rc / Cc + (dt / Cc) * (u * dOCVdT), dt / Rc / Cc],
        [0, 0, 0, dt / Rc / Cs, 1 - dt / Rc / Cs - dt / Ru / Cs]
    ],dtype=object)

    B = np.array([
        [-dt / (3600 * Q), 0, 0],
        [1 - A_RC, 0, 0],
        [0, A_HK - 1, 0],
        [(-M * hk + M0 * np.sign(u) + R * ir + R0 * u) * dt / Cc, 0, 0],
        [0, 0, dt / Ru / Cs]
    ],dtype=object)

    # 更新状态变量
    x_hat = np.dot(A, x) + np.dot(B, np.array([u, np.sign(u), Tf],dtype=object))

    return x_hat, v


def SOH_loss(tc, u):

    dt = 1
    B = 12934
    c = abs(u) / 10
    Ea = 31700 - 370.3 * c
    R = 8.31
    z = 0.55
    Ah = (20 / B / np.exp(-Ea / (R * tc))) ** (1 / z)
    delta_soh = - abs(u) * dt / (2*3600 * Ah)

    return delta_soh


def get_cccv(zk, ir, hk, v, temp):
    R = get_parameters('RParam', temp)
    eta = get_parameters('etaParam', temp)
    M = get_parameters('MParam', temp)
    M0 = get_parameters('M0Param', temp)
    R0 = get_parameters('R0Param', temp)

    OCV = OCV_fromSOCtemp(zk, temp, model_struct)
    u = (OCV + M * hk + M0 - R * ir - v) / R0
    u = eta * u

    return u

