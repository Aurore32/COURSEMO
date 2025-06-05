---
library_name: setfit
metrics:
- accuracy
pipeline_tag: text-classification
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: be to only wave point hearing which from human that by of normal with sound
    can range
- text: keep to two from wire used field becomes suitable when the contact copper
    doe away with not but metal magnetic thin remain strongly current right in help
    for taken diagram use and bar magnet across material is what keeper
- text: the fixed with some volume graph how gas container temperature of is trapped
    pressure in pa which
- text: the car speed graph how time of with
- text: to form americium an nucleus of neptunium
inference: true
model-index:
- name: SetFit
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: accuracy
      value: 0.9702970297029703
      name: Accuracy
---

# SetFit

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 5 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label   | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:--------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| thermal | <ul><li>'it student to freezing from lower thermometer occur the fixed doe point of not scale upper on about all decrease such dense correct evaporation and cold a near liquid temperature is check statement which'</li><li>'two rain water until do gradually have bed the move fixed after gas of not closely solid they are quickly about random state hit quite together remain at very something in apart straight and far a liquid storm is what matter'</li><li>'it bulb two where water melting block can must the so pure ice room that scale no metal on are boiling temperature at freezer marked in be diagram stem and cold thermometer is each'</li></ul> |
| em      | <ul><li>'steel to iron lead the nonferrous copper aluminium core electromagnet open group with only battery switch brass and connected an is which'</li><li>'divider student to circuit she row potential the ammeter of variable on resistance diagram and resistor an is determine what voltmeter which'</li><li>'it student to two four row potential find the one that of mistake they are resistance freely plastic current in negatively rod difference and given resistor both across measure an identical each which'</li></ul>                                                                                                                                   |
| nuclear | <ul><li>'and nucleus'</li><li>'atom the he diagram symbol helium an nucleus of which'</li><li>'minute the count rate per time at is what'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| general | <ul><li>'the in some to measured weight quantity and mass hot liquid cup of is what which'</li><li>'to weight mass used potential instrument gravity the internal of object energy on gravitational force compare an is what which'</li><li>'it to where upwards from row surface chemical heat example potential when the ball so into doe than after internal of not energy on less gravitational form all bounce ground stated converted strain and here way a hard an back is sound what which'</li></ul>                                                                                                                                                             |
| waves   | <ul><li>'of ray light'</li><li>'the strike diagram wave barrier and point water circular from row amplitude of coming which'</li><li>'to due from region the movement point than of wave greater amplitude air in diagram between and another sound which'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                      |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 0.9703   |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("the car speed graph how time of with")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 2   | 23.12  | 50  |

| Label   | Training Sample Count |
|:--------|:----------------------|
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| general | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| general | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| general | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| nuclear | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| general | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| general | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| nuclear | 10                    |
| general | 10                    |
| general | 10                    |
| nuclear | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| general | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| general | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| general | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| general | 10                    |
| nuclear | 10                    |
| general | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| general | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| general | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| waves   | 10                    |
| general | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| thermal | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| thermal | 10                    |
| em      | 10                    |
| em      | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| em      | 10                    |
| general | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| nuclear | 10                    |
| general | 10                    |
| em      | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| general | 10                    |
| thermal | 10                    |
| waves   | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| em      | 10                    |
| thermal | 10                    |
| general | 10                    |
| waves   | 10                    |

### Training Hyperparameters
- batch_size: (16, 2)
- num_epochs: (0.5, 16)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 0.008 | 1    | 0.1429        | -               |
| 0.016 | 2    | 0.0644        | -               |
| 0.032 | 4    | 0.1321        | -               |
| 0.048 | 6    | 0.099         | -               |
| 0.064 | 8    | 0.0178        | -               |
| 0.08  | 10   | 0.0253        | -               |
| 0.096 | 12   | 0.0168        | -               |
| 0.112 | 14   | 0.0145        | -               |
| 0.128 | 16   | 0.0059        | -               |
| 0.144 | 18   | 0.0268        | -               |
| 0.16  | 20   | 0.0368        | -               |
| 0.176 | 22   | 0.0029        | -               |
| 0.192 | 24   | 0.0301        | -               |
| 0.208 | 26   | 0.0221        | -               |
| 0.224 | 28   | 0.007         | -               |
| 0.24  | 30   | 0.0025        | -               |
| 0.256 | 32   | 0.0022        | -               |
| 0.272 | 34   | 0.0048        | -               |
| 0.288 | 36   | 0.0029        | -               |
| 0.304 | 38   | 0.0021        | -               |
| 0.32  | 40   | 0.0019        | -               |
| 0.336 | 42   | 0.0015        | -               |
| 0.352 | 44   | 0.0013        | -               |
| 0.368 | 46   | 0.0014        | -               |
| 0.384 | 48   | 0.0012        | -               |
| 0.4   | 50   | 0.001         | -               |
| 0.416 | 52   | 0.0012        | -               |
| 0.432 | 54   | 0.0015        | -               |
| 0.448 | 56   | 0.001         | -               |
| 0.464 | 58   | 0.0011        | -               |
| 0.48  | 60   | 0.0014        | -               |
| 0.496 | 62   | 0.0009        | -               |

### Framework Versions
- Python: 3.12.4
- SetFit: 1.1.1
- Sentence Transformers: 3.4.0
- Transformers: 4.44.2
- PyTorch: 2.5.1+cpu
- Datasets: 3.2.0
- Tokenizers: 0.19.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->