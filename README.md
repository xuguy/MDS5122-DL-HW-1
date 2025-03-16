### File structures and explanation:

./ <span style="color: rgba(0, 0, 0, 0.3)"> <------ project file: MDS5122-DL-HW1</span>
├── data/ </small> <span style="color: rgba(0, 0, 0, 0.3)"><------ cifar-10 and mnist for task A and B</span>
│   ├── cifar-10-batches-py/
│   └── MNIST/
│       └── raw/
├── dezero-master/ <span style="color: rgba(0, 0, 0, 0.3)"><------code for Task B</span>
│   └── dezero/ <span style="color: rgba(0, 0, 0, 0.3)"> <------ self-made framwork source code</span>
│       ├── MNISTdataset/ <span style="color: rgba(0, 0, 0, 0.3)"> <------ MNIST dataset for Task B, same in ../data</span>
│       │   └── __pycache__/
│       └── __pycache__/
├── kaggleFile/ <span style="color: rgba(0, 0, 0, 0.3)"> <------ Training codes for Task A, contains .ipynb for kaggle</span>
│   └── code/ <span style="color: rgba(0, 0, 0, 0.3)"> <------ model scource code</span>
│       └── mymodels/
│           └── __pycache__/
├── kaggleOutput/ <span style="color: rgba(0, 0, 0, 0.3)"> <------ outputs from kaggle of different experiment</span>
│   ├── bsln-1/ <span style="color: rgba(0, 0, 0, 0.3)"> <------ one file for each experiment</span>
│   ├── bsln-2/
│   ├── bsln-3/
│   ├── bsln-4/
│   ├── bsln-5/
│   ├── bsln-6/
│   ├── bsln-mnist/
│   ├── dezero-bsln/
│   ├── dezero-res18/
│   ├── res18/
│   ├── res18-mnist/
│   ├── res50/
│   └── vgg16/
└── texfile/ <span style="color: rgba(0, 0, 0, 0.3)"> <------ LaTeX file</span>
    └── fig/