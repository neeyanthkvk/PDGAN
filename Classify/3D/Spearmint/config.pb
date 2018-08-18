language: PYTHON
name:     "Spearmint"

variable {
 name: "conv"
 type: INT
 size: 4
 min:  1
 max:  64
}


variable {
  name: "dense"
  type: INT
  size: 1
  min:  4
  max:  256
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
