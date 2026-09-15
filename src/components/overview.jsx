import React from 'react';
import { render } from 'react-dom';

import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
marked.use(markedKatex({ throwOnError: false }));

export default class Overview extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-section">
        <a href="FIg2_fin.pdf" target="_blank" rel="noopener noreferrer" aria-label="Open framework figure PDF">
        <img
          src={`${this.props.teaser}`}
          className="uk-align-center uk-responsive-width"
          alt="PATH framework: object and contact belief, separate orientation belief, shared reference generation, and nonlinear inverse kinematics"
        />
        </a>
        <h2 className="uk-text-bold uk-heading-line uk-text-center">
          <span>Abstract</span>
        </h2>
        <div
          dangerouslySetInnerHTML={{
            __html: marked.parse(this.props.abstract),
          }}
        />
      </div>
    );
  }
}
