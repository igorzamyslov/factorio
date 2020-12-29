# factorio scripts
Contains useful scripts for factorio

## Requirements
- Python 3.7

## Running the scripts
```bash
python <script_name>
```

## Getting all the recipes
Paste and run the contents of the **dump_recipes.lua** 
from https://github.com/jmurrayufo/factorio/tree/master/dump_recipes 
in the factorio console.

The file will be available under `%APPDATA%\Factorio\script-output` (file: `recipe`)

It is possible to set priority on every recipe by adding `priority: true`.
It can be relevant when one item can be crafted in multiple ways.