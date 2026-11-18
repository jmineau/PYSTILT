{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{#- A dataclass field whose default is a function is not a method: _ext/api_pages.py. #}
.. auto{{ "attribute" if class_page.is_field(fullname) else objtype }}:: {{ objname }}
