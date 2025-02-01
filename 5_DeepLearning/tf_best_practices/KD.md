Teacher Model: Large, high-performing network that provides “knowledge” (logits, features, attention maps, or relational representations).
Student Model: Smaller, more efficient model trained to replicate teacher’s outputs or intermediate representations as closely as possible.

Distillation Techniques:
- Logit-based: Matching final-layer softened logits.
- Feature-based: Matching intermediate feature maps/activations.
- Relation-based: Matching pairwise or groupwise relationships.
- Self-distillation: A single model using its own deeper layers to guide shallower layers.
- Online distillation: Multiple students teach each other simultaneously.
- Multi-teacher distillation: Combining knowledge from several teachers.
- Task-specific distillation: Adapting KD to tasks like detection, segmentation, or machine translation.



# Knowledge Distillation Example

## Overview
Knowledge Distillation (KD) is a technique used to transfer knowledge from a large, high-performing model (Teacher) to a smaller, more efficient model (Student). This process allows the student model to achieve better accuracy than it would if trained solely on the ground-truth labels.

This document provides a step-by-step explanation of knowledge distillation applied to an image classification task.

## Hypothetical Example: Image Classification

### Teacher Model
- A **ResNet-50** pre-trained on ImageNet.
- Achieves **75% top-1 accuracy**.

### Student Model
- A **MobileNet** with fewer parameters.
- Faster inference but typically achieves **~70% accuracy** with naive training.

## Distillation Process

### 1. Forward Pass Through the Teacher
- Pass a training image through the **teacher model**.
- Compute the output **logits** and apply **softmax** with temperature \(T = 4\).
- The teacher outputs a probability distribution (soft targets) across 1,000 classes:
  
  ```
  p_teacher = [0.5, 0.2, 0.1, 0.05, 0.05, ...]  # Less spiky due to high temperature
  ```

### 2. Forward Pass Through the Student
- Pass the same image through the **student model**.
- Apply **softmax** with the same temperature (\(T = 4\)).
- The student produces its own probability distribution:
  
  ```
  p_student = [0.4, 0.25, 0.05, 0.1, ...]
  ```

### 3. Compute Losses
#### a) Distillation Loss
- Compare the student’s soft outputs with the teacher’s soft outputs using **Kullback-Leibler (KL) Divergence**:
  
  ```
  L_distill = KL(p_teacher, p_student)
  ```

#### b) Hard Label Loss (Optional but Common)
- Compute standard **cross-entropy loss** with the true labels:
  
  ```
  L_hard = CE(argmax(p_student), true_label)
  ```

### 4. Combine Losses
The final loss function is a weighted combination of the **hard label loss** and the **distillation loss**:

```
L = α * L_hard + (1 - α) * L_distill
```

where \( \alpha \) is a hyperparameter (e.g., \( \alpha = 0.5 \)) that balances between **ground-truth labels** and **teacher supervision**.

### 5. Optimize the Student Model
- Update the **student model's parameters** via backpropagation to minimize \(L\).
- Repeat for the entire dataset over multiple **epochs**.

## Outcome
- The student model typically **performs better** than if trained purely on the hard labels.
- Example: The **MobileNet** model, which originally achieved **~70% accuracy**, may now achieve **72-73% accuracy** thanks to knowledge distillation.

---

### References
- **Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network."**
- Knowledge Distillation implementations in **TensorFlow, PyTorch, and JAX**.

For more details, feel free to contribute or raise issues in this repository!

