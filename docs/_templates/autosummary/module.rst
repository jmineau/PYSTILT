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

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
