import React, { useState, useEffect } from 'react';
import { View, Image, StyleSheet } from 'react-native';

const SmartCropImage = ({ source, style, contentBounds }) => {
  const [imageStyle, setImageStyle] = useState({});

  useEffect(() => {
    if (contentBounds && source?.uri) {
      calculateOptimalDisplay(contentBounds);
    } else {
      // Default style if no bounds available
      setImageStyle({
        width: '100%',
        height: '100%',
      });
    }
  }, [contentBounds, source]);

  const calculateOptimalDisplay = (bounds) => {
    if (!bounds || bounds.minX === undefined) {
      setImageStyle({
        width: '100%',
        height: '100%',
      });
      return;
    }

    const { minX, maxX, minY, maxY, width, height } = bounds;
    
    // Calculate content dimensions
    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;
    
    if (contentWidth <= 0 || contentHeight <= 0) {
      setImageStyle({
        width: '100%',
        height: '100%',
      });
      return;
    }
    
    // Calculate content center
    const contentCenterX = (minX + maxX) / 2;
    const contentCenterY = (minY + maxY) / 2;
    
    // Calculate image center
    const imageCenterX = width / 2;
    const imageCenterY = height / 2;
    
    // Calculate scale to fit content in container with padding
    const containerPadding = 0.85; // Use 85% of container
    const scaleX = containerPadding / (contentWidth / width);
    const scaleY = containerPadding / (contentHeight / height);
    
    // Use smaller scale to ensure content fits
    const scale = Math.min(scaleX, scaleY, 3); // Cap at 3x zoom
    
    // Calculate translation to center the content
    const translateX = (imageCenterX - contentCenterX) * scale;
    const translateY = (imageCenterY - contentCenterY) * scale;
    
    setImageStyle({
      width: `${scale * 100}%`,
      height: `${scale * 100}%`,
      transform: [
        { translateX },
        { translateY }
      ],
    });
  };

  return (
    <View style={[styles.container, style]}>
      <Image
        source={source}
        style={[styles.baseImage, imageStyle]}
        resizeMode="contain"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  baseImage: {
    position: 'absolute',
  },
});

export default SmartCropImage;