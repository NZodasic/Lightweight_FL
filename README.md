The project format should follow:

project/
│
├── configs/
│   └── config.yaml
├── data/
│   └── data_loader.py   
│
├── models/
│   ├── resnet50.py
│   └── model_utils.py   
│
├── training/
│   ├── trainer.py
│   ├── evaluator.py
├── pruning/
│   ├── baseline.py
├── utils/                  # Logging, visualization, overfitting monitors, batchsize auto match with device
├── experiment.sh           # Convenience script to run the full pipeline
├── requirements.txt
└── EXPERIMENT/             # Auto-created run folders (logs, models, plots, …)
