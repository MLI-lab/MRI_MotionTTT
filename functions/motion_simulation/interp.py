
import torch
import numpy

def interpolate(input, width, kernel, coord, ndim,device):
    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = prod(pts_shape)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])
    output = torch.zeros([batch_size, npts], dtype=input.dtype, device=device)

    if ndim==2:
        output=_interpolate2(output, input, width, kernel, coord)
    elif ndim==3:
        output=_interpolate3(output, input, width, kernel, coord)

    return output.reshape(batch_shape + pts_shape)


def bilinear_interpolate_torch_gridsample(input, coord):
    coord=coord.unsqueeze(0).unsqueeze(0)
    tmp=torch.zeros_like(coord)
    tmp[:, :, :, 0] = ((coord[:, :, :, 1]+input.shape[2]/2) / (input.shape[2]-1))  # normalize to between  0 and 1
    tmp[:, :, :, 1] = ((coord[:, :, :, 0]+input.shape[2]/2) / (input.shape[2]-1)) # normalize to between  0 and 1
    tmp = tmp * 2 - 1  # normalize to between -1 and 1
    tmp=tmp.expand(input.shape[0],-1,-1,-1)
    return torch.nn.functional.grid_sample(input, tmp).squeeze(2)

def bilinear_interpolate_torch_gridsample_3d(input, coord):
    coord=coord.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    tmp=torch.zeros_like(coord)
    tmp[:, :, :,:, 0] = ((coord[:, :, :,:, 2]+input.shape[3]/2) / (input.shape[3]-1))  # normalize to between  0 and 1
    tmp[:, :, :,:, 1] = ((coord[:, :, :,:, 1]+input.shape[3]/2) / (input.shape[3]-1)) # normalize to between  0 and 1
    tmp[:, :, :,:, 2] = ((coord[:, :, :,:, 0]+input.shape[2]/2) / (input.shape[2]-1)) # normalize to between  0 and 1
    tmp = tmp * 2 - 1  # normalize to between -1 and 1
    tmp=tmp.expand(input.shape[0],-1,-1,-1,-1)
    return torch.nn.functional.grid_sample(input, tmp,padding_mode='border',align_corners=True).squeeze(2).squeeze(2)

def lin_interpolate(kernel, x):
    mask=torch.lt(x,1).float()
    x=x*mask
    n = len(kernel)
    idx = torch.floor(x * n)
    frac = x * n - idx
    left = kernel[idx.long()]
    mask2=torch.ne(idx,n-1).float()
    idx=idx*mask2
    right = kernel[idx.long() + 1]
    output=(1.0 - frac) * left + frac * right
    return output*mask*mask2


def _interpolate2(output, input, width, kernel, coord):
    batch_size, ny, nx = input.shape

    kx, ky = coord[:, -1], coord[:, -2]
    x0, y0 = (torch.ceil(kx - width / 2),
              torch.ceil(ky - width / 2))

    for y in range(int(width) + 1):
        wy = lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

        for x in range(int(width) + 1):
            w = wy * lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))

            yy=torch.fmod(y0+y,ny).long()
            xx=torch.fmod(x0+x,nx).long()
            output[:, :] = output[:, :] + w * input[:, yy, xx]

    return output

def _interpolate3(output, input, width, kernel, coord):
    batch_size, nz, ny, nx = input.shape

    kx, ky, kz = coord[:, -1], coord[:, -2], coord[:, -3]

    x0, y0, z0 = (torch.ceil(kx - width / 2),
                  torch.ceil(ky - width / 2),
                  torch.ceil(kz - width / 2))

    for z in range(int(width) + 1):
        wz = lin_interpolate(kernel, torch.abs(z0 + z - kz) / (width / 2))

        for y in range(int(width) + 1):
            wy = wz * lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

            for x in range(int(width) + 1):
                w = wy * lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))

                yy = torch.fmod(y0 + y, ny).long()
                xx = torch.fmod(x0 + x, nx).long()
                zz = torch.fmod(z0 + z, nz).long()
                output[:, :] = output[:, :] + w * input[:, zz, yy, xx]

    return output

def gridding(input, shape, width, kernel, coord, ndim, device):
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = prod(pts_shape)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = torch.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype, device=device)

    if ndim == 2:
        output=_gridding2(output, input, width, kernel, coord)
    elif ndim == 3:
        output=_gridding3(output, input, width, kernel, coord)

    return output.reshape(shape)

def _gridding2(output, input, width, kernel, coord):
    batch_size, ny, nx = output.shape

    kx, ky = coord[:, -1], coord[:, -2]

    x0, y0 = (torch.ceil(kx - width / 2),
              torch.ceil(ky - width / 2))

    for y in range(int(width) + 1):
        wy = lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

        for x in range(int(width) + 1):
            w = wy * lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))

            yy=torch.fmod(y0+y,ny).long()
            xx=torch.fmod(x0+x,nx).long()
            output[:, yy, xx] = output[:, yy, xx] + w * input[:, :]

    return output

def T_gridding3(output, input, width, kernel, coord):
    batch_size, nz, ny, nx = output.shape
    npts = coord.shape[0]

    for i in range(0,npts,1000):
        iend=min(i+1000,npts)
        kx, ky, kz = coord[i:iend, -1], coord[i:iend, -2], coord[i:iend, -3]

        x0, y0, z0 = (int(kx - width / 2)+1,
                      int(ky - width / 2)+1,
                      int(kz - width / 2)+1)

        x1, y1, z1 = (int(kx + width / 2)+1,
                      int(ky + width / 2)+1,
                      int(kz + width / 2)+1)

        for z in range(z0, z1 + 1):
            wz = lin_interpolate(kernel, abs(z - kz) / (width / 2))

            for y in range(y0, y1 + 1):
                wy = wz * lin_interpolate(kernel, abs(y - ky) / (width / 2))

                for x in range(x0, x1 + 1):
                    w = wy * lin_interpolate(kernel, abs(x - kx) / (width / 2))
                    output[:, z % nz, y % ny, x % nx][i:iend] += w * input[:, i:iend]

    return output

def TT_gridding3(output, input, width, kernel, coord):
    batch_size, nz, ny, nx = output.shape
    #kx, ky, kz = coord[:, -1], coord[:, -2], coord[:, -3]
    npts = coord.shape[0]
    for i in range(0,npts,10000):
        iend=min(i+1000,npts)
        kx, ky, kz = coord[i:iend, -1], coord[i:iend, -2], coord[i:iend, -3]
        x0, y0, z0 = (torch.ceil(kx - width / 2),
                      torch.ceil(ky - width / 2),
                      torch.ceil(kz - width / 2))

        for z in range(int(width) + 1):
            wz = lin_interpolate(kernel, torch.abs(z0 + z - kz) / (width / 2))

            for y in range(int(width) + 1):
                wy = wz * lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

                for x in range(int(width) + 1):
                    w = wy* lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))
                    yy = torch.fmod(y0 + y, ny).long()
                    xx = torch.fmod(x0 + x, nx).long()
                    zz = torch.fmod(z0 + z, nz).long()
                    try:
                        output[:, zz, yy, xx][i:iend]  += w * input[:, i:iend]
                    except:
                        output[:, zz, yy, xx] += w * input[:, i:iend]


    return output

def _gridding3(output, input, width, kernel, coord):
    batch_size, nz, ny, nx = output.shape

    kx, ky, kz = coord[:, -1], coord[:, -2], coord[:, -3]

    x0, y0, z0 = (torch.ceil(kx - width / 2),
                  torch.ceil(ky - width / 2),
                  torch.ceil(kz - width / 2))

    for z in range(int(width) + 1):
        wz = lin_interpolate(kernel, torch.abs(z0 + z - kz) / (width / 2))

        for y in range(int(width) + 1):
            wy = wz * lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

            for x in range(int(width) + 1):
                w = wy * lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))

                yy = torch.fmod(y0 + y, ny).long()
                xx = torch.fmod(x0 + x, nx).long()
                zz = torch.fmod(z0 + z, nz).long()
                output[:, zz, yy, xx] = output[:, zz, yy, xx] + w * input[:, :]

    return output

def prod(shape):
    """Computes product of shape.
    Args:
        shape (tuple or list): shape.
    Returns:
        Product.
    """
    return numpy.prod(shape)

def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)

def resize(input, oshape, ishift=None, oshift=None,device='cuda'):
    ishape_exp, oshape_exp = _expand_shapes(input.shape, oshape)

    if ishape_exp == oshape_exp:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    copy_shape = [min(i - si, o - so) for i, si, o,
                  so in zip(ishape_exp, ishift, oshape_exp, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape_exp, dtype=input.dtype, device=device)
    input = input.reshape(ishape_exp)
    output[oslice] = input[islice]

    return output.reshape(oshape)
