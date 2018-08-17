language: PYTHON
name:     "Spearmint"

variable {
 name: "conv"
 type: INT
 size: 2
 min:  0
 max:  1
}


variable {
  name: "dense"
  type: INT
  size: 5
  min:  -5
  max:  5
}

variable {
  name: "optimizer"
  type: ENUM
  size: 4
  options: "adam"
  options: "adagrad"
  options: "nadam"
  options: "SGD"
}
