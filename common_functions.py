import aioredis
import pickle as pc
import asyncio
import torch
import timeit
import numpy as np


async def multi_set_key_redis(keys, values):
    redis = await aioredis.create_redis_pool(
        'redis://localhost')
    async def transaction():
        tr = redis.multi_exec()
        for key, value in zip(keys, values):
            tr.set(key, value)
        await tr.execute()

    await transaction()
    redis.close()
    await redis.wait_closed()


async def multi_get_key_redis(keys):
    redis = await aioredis.create_redis_pool(
        'redis://localhost')
    result = []
    for key in keys:
        result.append(await redis.get(key))
    redis.close()
    await redis.wait_closed()
    # print(result)
    return result


# def push_params_redis(model):
#     i = -1
#     keys=[]
#     values=[]
#     for param in list(model.parameters()):
#         i = i+1
#         param_data = param.data.numpy()
#         p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
#         keys.append(i)
#         values.append(p)
#     asyncio.get_event_loop().run_until_complete(multi_set_key_redis(keys,values))
#
#
# def get_params_redis(shapes):
#     i = -1
#     params=[]
#     keys = []
#     for s in range(len(shapes)):
#         keys.append(s)
#     values = asyncio.get_event_loop().run_until_complete(multi_get_key_redis(keys))
#     for shape in shapes:
#         i = i + 1
#         param_np = pc._loads(values[i]).reshape(shape)
#         param_tensor = torch.nn.Parameter(torch.from_numpy(param_np))
#         params.append(param_tensor)
#     return params


def check_param_exists(model, db):
    if db.execute_command('exists', 'param_data0'):
        return True
    else:
        return False


def push_params_redis_init(model, db):
    i = -1
    pipe = db.pipeline()
    start_push_time = timeit.default_timer()
    for param in list(model.parameters()):
        i = i + 1
        param_data = param.data.numpy()
        # p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
        # param_data = param_data.flatten()
        param_shape = param_data.shape
        # param_data = param_data.tolist()
        # param_data.insert(0, param_shape[0])
        if len(param_shape) > 1:
            # param_data.insert(0, param_shape[1])
            mats = []
            loop_rec(param_data, param_shape, int(len(param_shape) - 2), 0, mats)
            for j, mat in enumerate(mats):
                mat = convert_mat_to_list(mat)
                # db.execute_command('ML.MATRIX.SET', 'param_data' + str(i) + str(j), *mat)
                pipe.execute_command('ML.MATRIX.SET', 'param_data' + str(i) + str(j), *mat)
        else:
            param_data = param_data.tolist()
            param_data.insert(0, param_shape[0])
            param_data.insert(0, 1)
            pipe.execute_command('ML.MATRIX.SET', 'param_data' + str(i), *param_data)
            # db.execute_command('ML.MATRIX.SET', 'param_data' + str(i), *param_data)
            # db.execute_command('ML.MATRIX.ADD', 'param_data', 'param_temp', 'param_data')
            # db.set(i, param_data)
    try:
        pipe.execute()
    except Exception as e:
        print(e)
    # print("pushed params : "+str(timeit.default_timer()-start_push_time))


def loop_rec(l, shape, n, i, k):
    if n >= 1:
        for x in range(shape[i]):
            loop_rec(l[x], shape, n - 1, i + 1, k)
    else:
        k.append(l)


def convert_mat_to_list(mat):
    mat = mat.flatten()
    mat_shape = mat.shape
    mat = mat.tolist()
    mat.insert(0, mat_shape[0])
    mat.insert(0, 1)
    return mat


def push_params_redis(params, db):
    i = -1
    pipe = db.pipeline()
    start_push_time = timeit.default_timer()
    for param in params:
        i = i + 1
        param_data = param.data.numpy()
        # p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
        # param_data = param_data.flatten()
        param_shape = param_data.shape
        # param_data = param_data.tolist()
        # param_data.insert(0, param_shape[0])
        if len(param_shape) > 1:
            # param_data.insert(0, param_shape[1])
            mats = []
            loop_rec(param_data, param_shape, int(len(param_shape) - 2), 0, mats)
            for j, mat in enumerate(mats):
                mat = convert_mat_to_list(mat)
                # db.execute_command('ML.MATRIX.SET', 'param_data' + str(i) + str(j), *mat)
                pipe.execute_command('ML.MATRIX.ADD', 'param_data' + str(i) + str(j),'param_data' + str(i) + str(j), *mat)
        else:
            param_data = param_data.tolist()
            param_data.insert(0, param_shape[0])
            param_data.insert(0, 1)
            pipe.execute_command('ML.MATRIX.ADD', 'param_data' + str(i),'param_data' + str(i), *param_data)
            # db.execute_command('ML.MATRIX.SET', 'param_data' + str(i), *param_data)
            # db.execute_command('ML.MATRIX.ADD', 'param_data', 'param_temp', 'param_data')
            # db.set(i, param_data)
    try:
        pipe.execute()
    except Exception as e:
        print(e)
    # print("pushed params : "+str(timeit.default_timer()-start_push_time))


def get_params_redis(db, shapes):
    i = -1
    params = []
    # start_get_time = timeit.default_timer()
    for shape in shapes:
        i = i + 1
        slices = []
        # print(shape)
        for slice in range(count_slices(shape)):
            if len(shape) >= 2:
                try:
                    param = db.execute_command('ML.MATRIX.GET', 'param_data' + str(i) + str(slice))
                except Exception as e:
                    print('ML.MATRIX.GET', 'param_data' + str(i) + str(slice))
                # print('param_data' + str(i) + str(slice))
                param = param[2:]
                slice = convert_list_to_mat(param, shape)
                # print(slice.shape)
                slices.append(slice)
            else:
                try:
                    param = db.execute_command('ML.MATRIX.GET', 'param_data' + str(i))
                except Exception as e:
                    print('ML.MATRIX.GET', 'param_data' + str(i))
                # print('param_data' + str(i))
                pshape = param[:2]
                param = param[2:]
                slice = np.reshape(param,pshape)
                # print(slice.shape)
                slices.append(slice)

        # layer = convert_slices_to_layer(np.array(slices), shape)
        # param_np = pc._loads(param).reshape(shape)
        # param_tensor = torch.nn.Parameter(torch.from_numpy(np.reshape(param, shape).astype(float)))
        # params.append(param_tensor.type(torch.FloatTensor))
        # slices.append(np.array(layer))
        # print(layer.shape)
        params.append(np.reshape(slices,shape).astype(float))
        # print('Get params time'+str(timeit.default_timer()-start_get_time))
    return params


def count_slices(shape):
    k = 1
    for i in range(len(shape) - 2):
        k = k * shape[i]
    return k


def convert_list_to_mat(param, shape):
    slice_shape = [shape[len(shape) - 2], shape[len(shape) - 1]]
    return np.reshape(param, slice_shape).astype(float)


def convert_slices_to_layer(mats, shape):
    # j = 0
    # z = []
    shape = shape[:len(shape)-2]

    # shape = shape[::-1]
    # for i in shape:
    #     temp = []
    #     for x in range(i):
    #         temp.append(np.array(mats[j]))
    #         j = j + 1
    #     z.append(np.array(temp))
    return np.reshape(mats, shape).astype(float)


# def push_params_redis_init(model, db):
#     i = -1
#     for param in list(model.parameters()):
#         i = i+1
#         param_data = param.data.numpy()
#         # p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
#         param_data = param_data.flatten()
#         param_shape = param_data.shape
#         param_data = param_data.tolist()
#         param_data.insert(0, param_shape[0])
#         if len(param_shape) > 1:
#             param_data.insert(0, param_shape[1])
#         else:
#             param_data.insert(0,1)
#         db.execute_command('ML.MATRIX.SET', 'param_data'+str(i), *param_data)
#         # db.execute_command('ML.MATRIX.ADD', 'param_data', 'param_temp', 'param_data')
#         # db.set(i, param_data)
#
#
# def push_params_redis(params, db):
#     i = -1
#     for param in params:
#         i = i+1
#         param_data = param.numpy()
#         # p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
#         param_data = param_data.flatten()
#         param_shape = param_data.shape
#         param_data = param_data.tolist()
#         param_data.insert(0, param_shape[0])
#         if len(param_shape) > 1:
#             param_data.insert(0, param_shape[1])
#         else:
#             param_data.insert(0,1)
#         # db.execute_command('ML.MATRIX.SET', 'param_temp'+str(i), *param_data)
#         # param_data.insert(0, 'param_data'+str(i))
#         db.execute_command('ML.MATRIX.ADD', 'param_data'+str(i), 'param_data'+str(i), *param_data)
#         # db.set(i, param_data)
#
#
# def get_params_redis(db, shapes):
#     i = -1
#     params=[]
#     for shape in shapes:
#         i = i + 1
#         param = db.execute_command('ML.MATRIX.GET','param_data'+str(i))
#         param = param[2:]
#         # param_np = pc._loads(param).reshape(shape)
#         # param_tensor = torch.nn.Parameter(torch.from_numpy(np.reshape(param, shape).astype(float)))
#         # params.append(param_tensor.type(torch.FloatTensor))
#         params.append(np.reshape(param, shape).astype(float))
#     return params
#


def get_shapes(model):
    shapes = []
    for param in list(model.parameters()):
        shapes.append(list(param.size()))
    return shapes


def set_params(optimizer, params):
    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            p.data = torch.from_numpy(params[i]).float()

# def push_params_memcache(model, client):
#     i = -1
#     for param in list(model.parameters()):
#         i = i + 1
#         param_data = param.data.numpy()
#         p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
#         client.set(str(i), p)


# def get_params_memcache(client, shapes):
#     i = -1
#     params=[]
#     for shape in shapes:
#         i = i + 1
#         param = client.get(str(i))
#         param_np = pc._loads(param).reshape(shape)
#         param_tensor = torch.nn.Parameter(torch.from_numpy(param_np))
#         params.append(param_tensor)
#     return params
