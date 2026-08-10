# DeepForest issues to file

---

## Issue 1: `on_fit_start` raises when `existing_train_dataloader` is set

**Title:** `on_fit_start` raises `AttributeError` when using `existing_train_dataloader`

**Body:**

When passing a custom dataloader via `existing_train_dataloader=` the guard in `on_fit_start` still raises:

```python
def on_fit_start(self):
    if self.config["train"]["csv_file"] is None:
        raise AttributeError("Cannot train with a train annotations file ...")
```

`config["train"]["csv_file"]` is `None` (correct — there is no CSV), but training should proceed because `existing_train_dataloader` is set. The check should be:

```python
if self.config["train"]["csv_file"] is None and self.existing_train_dataloader is None:
```

**Suggested fix:** change the condition in `on_fit_start` to also allow training when `existing_train_dataloader is not None`.

---

## Issue 2: `create_trainer` silently disables validation when `existing_val_dataloader` is set

**Title:** `create_trainer` disables validation even when `existing_val_dataloader` is provided

**Body:**

`create_trainer` decides whether to enable validation by checking only `config["validation"]["csv_file"]`:

```python
if not self.config["validation"]["csv_file"] is None:
    limit_val_batches = 1.0
    num_sanity_val_steps = 2
else:
    limit_val_batches = 0   # validation silently disabled
    num_sanity_val_steps = 0
```

If a user sets `existing_val_dataloader=` but leaves `config["validation"]["csv_file"]` as `None`, validation is silently disabled — no warning, no error, val metrics never appear. The condition should also check `self.existing_val_dataloader`:

```python
has_val = (self.config["validation"]["csv_file"] is not None
           or self.existing_val_dataloader is not None)
if has_val:
    limit_val_batches = 1.0
    num_sanity_val_steps = 2
else:
    limit_val_batches = 0
    num_sanity_val_steps = 0
```
