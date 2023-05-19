# hackathon-ing

Prepare dataset with jpg files:
```
python prepare_source_images.py path_to_original_dataset
```
Generate TensorFlow model:
```
python prepare_model.py
```
Generate .csv with predictions:
```
python generate_answers.py directory_with_images_to_classify
```
Classify singular image
```
python test_image.py path_to_image
```
