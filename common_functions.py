import aioredis
import pickle as pc
import asyncio
import torch
import numpy as np


async def multi_set_key_redis(keys,values):
    redis = await aioredis.create_redis_pool(
        'redis://localhost')

    async def transaction():
        tr = redis.multi_exec()
        for key, value in zip(keys, values):
            tr.set(key,value)
        await tr.execute()
    await transaction()
    redis.close()
    await redis.wait_closed()


async def multi_get_key_redis(keys):
    redis = await aioredis.create_redis_pool(
        'redis://localhost')
    result =[]
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


def push_params_redis_init(model, db):
    i = -1
    for param in list(model.parameters()):
        i = i+1
        param_data = param.data.numpy()
        # p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
        param_data = param_data.flatten()
        param_shape = param_data.shape
        param_data = param_data.tolist()
        param_data.insert(0, param_shape[0])
        if len(param_shape) > 1:
            param_data.insert(0, param_shape[1])
        else:
            param_data.insert(0,1)
        db.execute_command('ML.MATRIX.SET', 'param_data'+str(i), *param_data)
        # db.execute_command('ML.MATRIX.ADD', 'param_data', 'param_temp', 'param_data')
        # db.set(i, param_data)


def push_params_redis(optimizer, db):
    i = -1
    for group in optimizer.param_groups:
        for param in group['params']:
            i = i+1
            param_data = param.grad.data.numpy()
            # p = pc._dumps(param_data, protocol=pc.HIGHEST_PROTOCOL)
            param_data = param_data.flatten()
            param_shape = param_data.shape
            param_data = param_data.tolist()
            param_data.insert(0, param_shape[0])
            if len(param_shape) > 1:
                param_data.insert(0, param_shape[1])
            else:
                param_data.insert(0,1)
            db.execute_command('ML.MATRIX.SET', 'param_temp'+str(i), *param_data)
            db.execute_command('ML.MATRIX.ADD', 'param_data'+str(i), 'param_temp'+str(i), 'param_data'+str(i))
            # db.set(i, param_data)


def get_params_redis(db, shapes):
    i = -1
    params=[]
    for shape in shapes:
        i = i + 1
        param = db.execute_command('ML.MATRIX.GET','param_data'+str(i))
        param = param[2:]
        # param_np = pc._loads(param).reshape(shape)
        # param_tensor = torch.nn.Parameter(torch.from_numpy(np.reshape(param, shape).astype(float)))
        # params.append(param_tensor.type(torch.FloatTensor))
        params.append(np.reshape(param, shape).astype(float))
    return params


def get_shapes(model):
    shapes = []
    for param in list(model.parameters()):
        shapes.append(list(param.size()))
    return shapes


def set_params(optimizer, params):
    optimizer.step(grads=params)


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
