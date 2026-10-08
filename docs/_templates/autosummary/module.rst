{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{#- Tables of what the module defines, each item on its own page. #}
{% for title, items in [("Attributes", attributes), ("Functions", functions), ("Classes", classes), ("Exceptions", exceptions)] %}
{% if items %}
.. rubric:: {{ title }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in items %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endfor %}

{#- Public names the package re-exports from its submodules: _ext/api_pages.py. #}
{% set exported = module_page.exported(fullname) %}
{% if exported %}
.. rubric:: Exported from submodules

.. autosummary::
   :nosignatures:
{% for item in exported %}
   ~{{ item }}
{%- endfor %}
{% endif %}

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
