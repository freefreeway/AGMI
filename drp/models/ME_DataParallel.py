from torch_geometric.nn.data_parallel import DataParallel
import torch


class MEDataParallel(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None,
                 follow_batch=[], exclude_keys=[]):
        super(MEDataParallel, self).__init__(module, device_ids, output_device, follow_batch, exclude_keys)

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        print(*inputs, **kwargs)
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        """Validation step function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for ``scatter_kwargs``.
        """
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output

    def update_encoder_buffer(self, batch, cell_edges_attr, cell_edges_index, num_genes_nodes):
        self.module.update_encoder_buffer(batch, cell_edges_attr, cell_edges_index, num_genes_nodes)
