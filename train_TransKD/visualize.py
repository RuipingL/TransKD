import numpy as np

from torch.autograd import Variable

from visdom import Visdom

class Dashboard:

    def __init__(self, port):
        self.vis = Visdom(port=port)

    def loss(self, losses, title):
        x = np.arange(1, len(losses)+1, 1)

        self.vis.line(losses, x, env='loss', opts=dict(title=title))

    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        self.vis.image(image, env='images', opts=dict(title=title))
    def add_scalar(self, win, x, y, opts=None, trace_name=None):
        """ Draw line
        """
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        default_opts = {'title': win}
        if opts is not None:
            default_opts.update(opts)
        update = 'append' if win is not None else None
        self.vis.line(X=x, Y=y, opts=default_opts, win=win, env='main', update=update, name=trace_name)
    def add_doubleline(self, epoch, val_loss, train_loss, title, win):
        self.vis.line(
            X=np.column_stack((epoch, epoch)),
            Y=np.column_stack((train_loss, val_loss)),
            opts=dict(legend=["Training Loss", "Validation Loss"],
                    showlegend=True,
                    markers=False,
                    title=title,
                    xlabel='epoch',
                    ylabel='',
                    fillarea=False),
            update= 'append',
            win= win,
            env= 'main'
        )