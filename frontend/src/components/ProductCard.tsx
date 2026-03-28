import React from 'react';
import productsData from '../data/products.json';

interface Props {
  intent: string;
}

export const ProductCard: React.FC<Props> = ({ intent }) => {
  const productInfo = (productsData as Record<string, any>)[intent];

  if (!productInfo) return null;

  return (
    <div className="product-card glass-panel">
      <h3>📋 {productInfo.name}</h3>
      <ul>
        {Object.entries(productInfo).map(([key, value]) => {
          if (key === 'name') return null;
          return (
            <li key={key}>
              <span className="card-key">{key.replace('_', ' ')}:</span>
              <span className="card-value">{value as string}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
};