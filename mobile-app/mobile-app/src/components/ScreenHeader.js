import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Platform,
} from 'react-native';

export default function ScreenHeader({ 
  title, 
  onBack, 
  showBack = true,
  rightIcon,
  onRightPress,
  titleStyle,
  largeTitle = false 
}) {
  return (
    <View style={styles.header}>
      {showBack ? (
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Text style={styles.backText}>‹</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.headerLeft} />
      )}
      
      <Text style={[
        largeTitle ? styles.largeTitle : styles.headerTitle,
        titleStyle
      ]}>
        {title}
      </Text>
      
      {rightIcon ? (
        <TouchableOpacity onPress={onRightPress} style={styles.rightButton}>
          {typeof rightIcon === 'string' ? (
            <Text style={styles.rightIcon}>{rightIcon}</Text>
          ) : (
            rightIcon
          )}
        </TouchableOpacity>
      ) : (
        <View style={styles.headerRight} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 10,
    paddingHorizontal: 20,
  },
  headerLeft: {
    width: 40,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backText: {
    fontSize: 32,
    color: '#FFFFFF',
    fontWeight: '200',
  },
  headerTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: '#FFFFFF',
    flex: 1,
    textAlign: 'center',
  },
  largeTitle: {
    fontSize: 34,
    fontWeight: '700',
    color: '#FFFFFF',
    flex: 1,
    textAlign: 'left',
  },
  headerRight: {
    width: 40,
  },
  rightButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  rightIcon: {
    fontSize: 20,
    color: '#FFFFFF',
  },
});