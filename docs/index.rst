.. SAM PyTorch documentation master file, created by
   sphinx-quickstart on Fri Feb 26 13:32:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SAM PyTorch's documentation!
=======================================

*********
Install
*********
Stable release

.. code-block:: console

   pip3 install sam-pytorch

Latest code

.. code-block:: console

   pip3 install git+https://github.com/tourdeml/sam

******************
Sample usage
******************

.. code-block:: python

   model = resnet18()
   optim = torch.optim.SGD(model.parameters(), 1e-3)
   optim = SAM(model.parameters(), optim)
   def closure():
      optim.zero_grad()
      loss = model(torch.randn(1,3,64,64)).sum()
      loss.backward()
      return loss
   optim.step(closure)

.. toctree::
   :maxdepth: 2
   :caption: API reference

   sam
