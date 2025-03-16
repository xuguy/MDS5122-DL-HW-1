### 文件结构
- 总目录kaggleFile
- 子目录：
  - code: 用于存放模型文件mymodels，该文件同时也是一个可import的python package
    - mymodels
      - mocdels.py
      - __init__.py
    - kernel-metadata.json: 向kaggle打包代码文件的metadata
  - main.ipynb: 总控接口
  - README.md

# kaggle 命令：
- 工作路径： `D:\BaiduNetDisk\BaiduSyncdisk\CUHKsz\course2\DeepL\dl-assign1\kaggleFile`
- 若模型文件有改动：`kaggle datasets version -p ./code --dir-mode zip -m "datasets v1"`
- 若main.py有改动：kaggle kernels push -p .
- 改动后便捷上传测试：`kaggle datasets version -p ./code --dir-mode zip -m "no comment";  kaggle kernels push -p . `
  - kaggle datasets version -p ./code --dir-mode zip -m "no comment";  kaggle kernels push -p . 