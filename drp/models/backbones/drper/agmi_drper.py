import torch.nn
from drp.models.registry import BACKBONES
from drp.models.builder import build_component

@BACKBONES.register_module()
class AGMIDRPer(torch.nn.Module):

    def __init__(self,
                 drug_encoder,
                 genes_encoder,
                 neck,
                 head):
        super().__init__()

        # body
        self.drug_encoder = build_component(drug_encoder)
        self.genes_encoder = build_component(genes_encoder)
        self.head = build_component(head)
        self.neck = build_component(neck)

    def forward(self, data):
        """Forward function.

        Args:
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
            :param test_mode:
            :param gt:
            :param data:
        """
        x_cell = data.x_cell
        _, channels = x_cell.shape
        x_d, x_d_edge_index, x_d_batch = \
            data.x, data.edge_index, data.batch
        batch_size = x_d_batch.max().item() + 1

        x_cell = x_cell.view(-1, channels)
        g_embed = self.genes_encoder(x_cell)
        g_embed = g_embed.view(batch_size, -1, channels).permute(0, 2, 1)
        g_embed = self.neck(g_embed)

        drug_embed = self.drug_encoder(x_d, x_d_edge_index, x_d_batch)

        output = self.head(drug_embed, g_embed)

        return output
