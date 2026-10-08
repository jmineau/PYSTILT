{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{#- Which members get a row and a page of their own, and what is inherited:
    _ext/api_pages.py. Pydantic's own members are left out there, so no list
    of them is kept here. #}
{% set attrs = class_page.members(fullname, attributes) | sort(case_sensitive=False) %}
{% set meths = class_page.members(fullname, all_methods) | sort(case_sensitive=False) %}

{% if class_page.is_enum(fullname) %}
.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
{% else %}
.. class-signature:: {{ fullname }}

{% if objname.endswith('Config') or objname.endswith('GeometrySpec') or objname in ('Bounds', 'Grid', 'Mesh') or objname.endswith('Receptor') %}
Parameters
----------

.. config-model:: {{ fullname }}
{% else %}
.. class-parameters:: {{ fullname }}
{% endif %}

{% if attrs %}
Attributes
----------

.. autosummary::
   :toctree:
{% for item in attrs %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}

{% if meths %}
Methods
-------

.. autosummary::
   :toctree:
{% for item in meths %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}

{% for base, members in class_page.inherited(fullname, attributes + all_methods) %}
{% if loop.first %}
Inherited
---------

{% endif %}
From :class:`~.{{ base }}`:
{%- for member in members %} :py:obj:`~.{{ member }}`{{ "," if not loop.last }}{% endfor %}

{% endfor %}
{% endif %}
